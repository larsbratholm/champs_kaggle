import os
import sys
os.environ['OMP_NUM_THREADS'] = '8'
import os
import math
import torch
import db
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import cust_model
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader
from pytorch_transformers import BertConfig , AdamW, WarmupLinearSchedule
from optimizer import RAdam

import warnings
warnings.filterwarnings(action='ignore')

import argparse
import logging

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Get script location
script_dir = os.path.abspath(os.path.dirname(__file__))

SOURCE_PATH = script_dir + '../../../../data/xyz/'
FINETUNED_MODEL_PATH = script_dir + '../../../models/BERT-based/v25_2'


class CFG:
    learning_rate=3.0e-04
    batch_size=128
    num_workers=2
    print_freq=100
    test_freq=1
    start_epoch=0
    num_train_epochs=80
    max_seq_length=135
    warmup_steps=30
    max_grad_norm=1000
    gradient_accumulation_steps=1
    weight_decay=0.01
    hidden_size=832
    num_hidden_layers=8
    num_attention_heads=8
    intermediate_size=2048
    dropout=0.1        
CFG.batch_size = CFG.batch_size // CFG.gradient_accumulation_steps


def main():    
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", type=str, default='')    
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--nepochs", type=int, default=CFG.num_train_epochs)    
    parser.add_argument("--wsteps", type=int, default=CFG.warmup_steps)
    parser.add_argument("--nlayers", type=int, default=CFG.num_hidden_layers)
    parser.add_argument("--nahs", type=int, default=CFG.num_attention_heads)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--dropout", type=float, default=CFG.dropout)
    parser.add_argument("--types", nargs='+', type=str, 
                        default=['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN'], 
                        help='3JHC,2JHC,1JHC,3JHH,2JHH,3JHN,2JHN,1JHN')
    parser.add_argument("--train_file", default="train_mute_cp")
    parser.add_argument("--test_file", default="test_mute_cp")
    parser.add_argument("--pseudo_path", default="")
    parser.add_argument("--pseudo", action='store_true')
    parser.add_argument("--gen_pseudo", action='store_true')
    parser.add_argument("--use_all", action='store_true')
    parser.add_argument("--structure_file", default="structures_mu")
    parser.add_argument("--contribution_file", default="scalar_coupling_contributions")        
    args = parser.parse_args()
    print(args) 
    
    CFG.batch_size=args.batch_size
    CFG.num_train_epochs=args.nepochs
    CFG.warmup_steps=args.wsteps
    CFG.num_hidden_layers=args.nlayers
    CFG.num_attention_heads=args.nahs
    CFG.learning_rate=args.lr
    CFG.dropout=args.dropout
    CFG.seed =  args.seed
    print(CFG.__dict__)
    
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    
    #if not args.eval:    
    if True:
        train_df = load_csv(args.train_file)
        
        structures_df = load_csv(args.structure_file)  
        structures_df[['x', 'y', 'z']] -= structures_df.groupby('molecule_name')[['x', 'y', 'z']].transform('mean')        
        
        contributions_df = load_csv(args.contribution_file)
        train_df = train_df.merge(contributions_df, how='left')   
        train_df = normalize_cols(train_df, ['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso'])        
        train_df = add_extra_features(train_df, structures_df)
        train_df = train_df.fillna(1e08)
        n_mols = train_df['molecule_name'].nunique()
        train_df, valid_df = train_test_split(train_df, 5000 )
        
        # only molecules with the args.types
        print(train_df['molecule_name'].nunique())
        mol_names_with_at = train_df[train_df['type'].isin(args.types)]['molecule_name'].unique()
        train_df = train_df[train_df['molecule_name'].isin(mol_names_with_at)].reset_index(drop=True)
        print(train_df['molecule_name'].nunique())
        
        # Print the 5 rows of valid_df to verify whether the valid_df is the same as the previous experiment.
        print(valid_df.head(5))
        
        if args.pseudo:        
            test_df = load_csv(args.test_file)
            logger.info(f'loading dataset - {args.pseudo_path} ...')
            test_pseudo_df = pd.read_csv(args.pseudo_path)
            #mol_names_jhn = train_df[test_df['type'].isin(['1JHN', '2JHN', '3JHN'])]['molecule_name'].unique()
            #test_df = test_df[test_df['molecule_name'].isin(mol_names_jhn)].reset_index(drop=True)        
            test_df = add_extra_features(test_df, structures_df)
            test_df = test_df.set_index('id')
            test_pseudo_df = test_pseudo_df.set_index('id')
            test_df[['scalar_coupling_constant',  'fc', 'sd', 'pso', 'dso']] = test_pseudo_df[['scalar_coupling_constant',  'fc', 'sd', 'pso', 'dso']]
            test_df = test_df.reset_index()            
            #test_df = normalize_target(test_df)
            test_df = normalize_cols(test_df, ['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso'])
            #test_df = test_df.assign(fc=1e08, sd=1e08, pso=1e08, dso=1e08)
            train_df['weight'] = 1.0
            valid_df['weight'] = 1.0
            test_df['weight'] = 1.0
            n_mols = test_df['molecule_name'].nunique()            
            train_df = train_df.append(test_df).reset_index(drop=True)
        else:
            train_df['weight'] = 1.0
            valid_df['weight'] = 1.0
        
        if args.use_all:
            train_df = train_df.append(valid_df) 
        
        print(f' n_train:{len(train_df)}, n_valid:{len(valid_df)}')
    
    config = BertConfig(            
            3, # not used
            hidden_size=CFG.hidden_size,
            num_hidden_layers=CFG.num_hidden_layers,
            num_attention_heads=CFG.num_attention_heads,
            intermediate_size=CFG.intermediate_size,
            hidden_dropout_prob=CFG.dropout,
            attention_probs_dropout_prob=CFG.dropout,
        )    
    model = cust_model.SelfAttn(config)
    if args.model != "":
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        CFG.start_epoch = checkpoint['epoch']        
        model.load_state_dict(checkpoint['state_dict'])        
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.model, checkpoint['epoch']))
    model.cuda()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('parameters: ', count_parameters(model))
    
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # to produce the submission.csv
    if args.eval:
        test_df = load_csv(args.test_file)
        structures_df = load_csv(args.structure_file)
        structures_df[['x', 'y', 'z']] -= structures_df.groupby('molecule_name')[['x', 'y', 'z']].transform('mean')        
        test_df = add_extra_features(test_df, structures_df)
        test_df = test_df.assign(fc=1e08, sd=1e08, pso=1e08, dso=1e08) 
        test_df['scalar_coupling_constant'] = 0
        test_df['weight'] = 1.0
        test_db = db.MolDB(test_df, CFG.max_seq_length)
        test_loader = DataLoader(
            test_db, batch_size=CFG.batch_size, shuffle=False,
            num_workers=CFG.num_workers)
        res_df = validate(test_loader, model, args.types)        
        res_df = unnormalize_cols(res_df, cols=['fc', 'sd', 'pso', 'dso'])
        res_df = unnormalize_target(res_df, 'prediction1')
        if args.gen_pseudo:
            res_df['scalar_coupling_constant'] = res_df['prediction1']
            res_df = res_df[res_df['id']>-1].sort_values('id')
            res_df[['id', 'scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso']].to_csv(f'pseudo_{CFG.seed}.csv', index=False)
            return
        res_df['prediction4']= res_df[['fc', 'sd', 'pso', 'dso']].sum(1)
        res_df['prediction']= res_df[['prediction1','prediction4']].mean(1)        
        res_df['scalar_coupling_constant'] = res_df['prediction']
        res_df = res_df[res_df['id']>-1].sort_values('id')
        os.makedirs('output', exist_ok=True)
        res_df[['id', 'scalar_coupling_constant']].to_csv(f'output/submission_{CFG.seed}.csv', index=False)        
        return
    
    train_db = db.MolDB(train_df, CFG.max_seq_length)    
    print('preloading dataset ...')
    train_db = db.MolDB_FromDB(train_db, 10)    
    valid_db = db.MolDB(valid_df, CFG.max_seq_length)    
    num_train_optimization_steps = int(
        len(train_db) / CFG.batch_size / CFG.gradient_accumulation_steps) * (CFG.num_train_epochs-CFG.start_epoch)
    print('num_train_optimization_steps', num_train_optimization_steps)      

    train_loader = DataLoader(
        train_db, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True)
    val_loader = DataLoader(
        valid_db, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers)
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                           lr=CFG.learning_rate,
                           weight_decay=CFG.weight_decay,                           
                           )
    scheduler = WarmupLinearSchedule(optimizer, CFG.warmup_steps,
                                        t_total=num_train_optimization_steps
                                     )
    
    def get_lr():
        return scheduler.get_lr()[0]
    
    if args.model != "":
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = CFG.learning_rate
        mae_log_df = checkpoint['mae_log']
        del checkpoint
    else:
        mae_log_df = pd.DataFrame(columns=(['EPOCH']+['LR']+args.types + ['OVERALL']) )     
    os.makedirs('log', exist_ok=True)
    
    
    res_df = validate(val_loader, model, args.types)        
    res_df = unnormalize_cols(res_df, cols=['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso'])
    res_df = unnormalize_target(res_df, 'prediction1')            
    res_df['prediction4']= res_df[['fc', 'sd', 'pso', 'dso']].sum(1)
    res_df['prediction']= res_df[['prediction1','prediction4']].mean(1)
    res_df.to_csv(f'log/valid_df_{"_".join(args.types)}.csv', index=False)
    overall_mae, maes = metric(res_df, args.types)
    print(overall_mae, maes)    
    
    
    curr_lr = get_lr()
    print(f'initial learning rate:{curr_lr}')
    for epoch in range(CFG.start_epoch, CFG.num_train_epochs):
        # train for one epoch
                
        #print(adjust_learning_rate(optimizer, epoch))    
        train(train_loader, model, optimizer, epoch, args.types, scheduler)
       
        if epoch % CFG.test_freq == 0:
            res_df = validate(val_loader, model, args.types)        
            res_df = unnormalize_cols(res_df, cols=['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso'])
            res_df = unnormalize_target(res_df, 'prediction1')            
            res_df['prediction4']= res_df[['fc', 'sd', 'pso', 'dso']].sum(1)
            res_df['prediction']= res_df[['prediction1','prediction4']].mean(1)
            res_df.to_csv(f'log/valid_df_{"_".join(args.types)}.csv', index=False)
            overall_mae, maes = metric(res_df, args.types)
            
            # write log file
            mae_row = dict([(typ, [mae]) for typ, mae in maes.items() if typ in args.types])
            mae_row.update({'EPOCH':(epoch),'OVERALL':overall_mae, 'LR':curr_lr})
            mae_log_df = mae_log_df.append(pd.DataFrame(mae_row), sort=False)
            print(mae_log_df.tail(20))        
            mae_log_df.to_csv(f'log/{"_".join(args.types)}.csv', index=False)
            
            #scheduler.step(overall_mae)
            curr_lr = get_lr()
            print(f'set the learning_rate: {curr_lr}')
            
            # evaluate on validation set
            batch_size = CFG.batch_size            
            pseudo_path = '' if not args.pseudo else '_' + args.pseudo_path 
            curr_model_name = (f'b{batch_size}_l{config.num_hidden_layers}_'
                               f'mh{config.num_attention_heads}_h{config.hidden_size}_'
                               f'd{CFG.dropout}_'
                               f'ep{epoch}_{"_".join(args.types)}_s{CFG.seed}{pseudo_path}.pt')
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the cust_model it-self    
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'transformer',
                'state_dict': model_to_save.state_dict(),
                'mae_log': mae_log_df,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                },
                FINETUNED_MODEL_PATH, curr_model_name
            )                                                
                                         
    print('done')
    
def train(train_loader, model, optimizer, epoch, bond_types, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #aux_losses = AverageMeter()
    sent_count = AverageMeter()
    
    # switch to train mode
    model.train()
    
    preds = []

    start = end = time.time()
    global_step = 0
    bond_types = [db.type2idx[bond_type] for bond_type in bond_types]
    for step, (atom0, atom1, typ, xyz0, xyz1, mu0, mu1, mask, target, weight, length) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        atom0, atom1, typ, xyz0, xyz1, mu0, mu1, mask, target, weight = (
            atom0.cuda(), atom1.cuda(), typ.cuda(), xyz0.cuda(), xyz1.cuda(), 
            mu0.cuda(), mu1.cuda(), mask.cuda(), target.cuda(), weight.cuda())
        
        max_len = length.max()        
        atom0 = atom0[:, :max_len].contiguous()
        atom1 = atom1[:, :max_len].contiguous()
        typ = typ[:, :max_len].contiguous()
        xyz0 = xyz0[:, :max_len].contiguous()
        xyz1 = xyz1[:, :max_len].contiguous()
        mu0 = mu0[:, :max_len].contiguous()
        mu1 = mu1[:, :max_len].contiguous()
        mask = mask[:, :max_len].contiguous()
        target = target[:, :max_len].contiguous()
        weight = weight[:, :max_len].contiguous()
        
        batch_size = atom0.size(0)
        
        # compute loss
        pred1, pred5 = model(atom0, atom1, typ, xyz0, xyz1, mu0, mu1, mask)
        mae_loss = torch.abs((pred5-target)*(target < 1e08).float() ).sum(-1).view(-1) * weight.view(-1)
        # consider only the difference less than 1e08
        mae_loss = mae_loss[mask.view(-1)]
        typ = typ.view(-1)[mask.view(-1)]
        
        loss = 0
        for bond_type in bond_types: 
            loss += torch.log(mae_loss[typ == bond_type].mean())              
        loss /= len(bond_types)            
         
        # record loss
        losses.update(loss.item(), batch_size)
        
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        if (step + 1) % CFG.gradient_accumulation_steps == 0:      
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        if step % CFG.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})  '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr}  '
                  'sent/s {sent_s:.0f} '
                  .format(
                   epoch, step, len(train_loader), batch_time=batch_time,                   
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   lr=scheduler.get_lr()[0],
                   sent_s=sent_count.avg/batch_time.avg
                   ))


def validate(valid_loader, model, bond_types):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #aux_losses = AverageMeter()
    sent_count = AverageMeter() 

    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    
    preds = []
    bond_types = [db.type2idx[bond_type] for bond_type in bond_types]
    for step, (atom0, atom1, typ, xyz0, xyz1, mu0, mu1, mask, target, weight, length) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        atom0, atom1, typ, xyz0, xyz1, mu0, mu1, mask, target, weight = (
            atom0.cuda(), atom1.cuda(), typ.cuda(), xyz0.cuda(), xyz1.cuda(), 
            mu0.cuda(), mu1.cuda(), mask.cuda(), target.cuda(), weight.cuda())
        
        max_len = length.max()        
        atom0 = atom0[:, :max_len].contiguous()
        atom1 = atom1[:, :max_len].contiguous()
        typ = typ[:, :max_len].contiguous()
        xyz0 = xyz0[:, :max_len].contiguous()
        xyz1 = xyz1[:, :max_len].contiguous()
        mu0 = mu0[:, :max_len].contiguous()
        mu1 = mu1[:, :max_len].contiguous()
        mask = mask[:, :max_len].contiguous()
        target = target[:, :max_len].contiguous()
        weight = weight[:, :max_len].contiguous()
        
        batch_size = atom0.size(0)     
        
        # compute loss
        with torch.no_grad():
            pred1, pred5 = model(atom0, atom1, typ, xyz0, xyz1, mu0, mu1, mask)
            mae_loss = torch.abs((pred5-target)*(target < 1e08).float() ).sum(-1).view(-1) * weight.view(-1)                
            mae_loss = mae_loss[mask.view(-1)]
            
            typ = typ.view(-1)[mask.view(-1)]
            
            loss = 0
            for bond_type in bond_types: 
                loss += torch.log(mae_loss[typ == bond_type].mean())                    
            loss /= len(bond_types)            
        
            preds.append(pred5.view(-1, 5)[mask.view(-1).byte()].float().cpu())            
             
        # record loss
        losses.update(loss.item(), batch_size)
        sent_count.update(batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):          
            print('Valid: [{0}][{1}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '                  
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'sent/s {sent_s:.0f}'
                  .format(
                   step, len(valid_loader), batch_time=batch_time,                   
                   data_time=data_time, loss=losses,
                   sent_s=sent_count.avg/batch_time.avg
                   ))
    preds = torch.cat(preds).numpy()    
    valid_df = valid_loader.dataset.df.copy()
    preds = pd.DataFrame(preds, columns=['prediction1', 'fc', 'sd', 'pso', 'dso'])
    valid_df[['prediction1', 'fc', 'sd', 'pso', 'dso']] = preds
    return valid_df


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()


def load_csv(filename):  
    logger.info(f'loading dataset - {filename}.csv ...')
    csv_filename = os.path.join(SOURCE_PATH, filename+'.csv')
    df = pd.read_csv(csv_filename)
    #if 'test' in filename:
    #    df['target'] = -1 
    return df


def add_extra_features(df, extra_features):
    logger.info('adding extra features ...')
    feature_cols = extra_features.columns[2:]
    left = df.merge(extra_features, how='left',
                   left_on=['molecule_name', 'atom_index_0'], 
                   right_on=['molecule_name', 'atom_index'])[feature_cols].add_suffix('_0')
    right = df.merge(extra_features, how='left',
                   left_on=['molecule_name', 'atom_index_1'], 
                   right_on=['molecule_name', 'atom_index'])[feature_cols].add_suffix('_1')    
    df = pd.concat([df, left, right], axis=1)    
    return df

def add_extra_mol_features(df, extra_features):    
    df = df.merge(extra_features, how='left',
                   left_on=['molecule_name'], 
                   right_on=['molecule_name'])
    return df


def train_test_split(df, n_valid=5000):
    groups = df.groupby('molecule_name').groups
    molecule_names = list(groups.keys())
    random.shuffle(molecule_names)
    train_df = df[df['molecule_name'].isin(molecule_names[:-n_valid])].reset_index(drop=True)
    valid_df = df[df['molecule_name'].isin(molecule_names[-n_valid:])].reset_index(drop=True)
    return train_df, valid_df


def metric(df, bond_types):
    maes = {}
    for t in bond_types:
        y_true = df[df.type==t].scalar_coupling_constant.values
        y_pred = df[df.type==t].prediction.values
        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))
        maes[t] = mae
    return np.mean(list(maes.values())), maes 


def save_checkpoint(state, model_path, model_filename):
    print('saving cust_model ...')
    if not os.path.exists(model_path):
        os.makedirs(model_path)    
    torch.save(state, os.path.join(model_path, model_filename))


COUPLING_TYPE_STATS={
    #type   #mean, std, min, max
    '1JHC':  [94.9761528641869,   18.27722399839607,   66.6008,   204.8800],
    '2JHC':  [-0.2706244378832,    4.52360876732858,  -36.2186,    42.8192],
    '3JHC':  [3.6884695895355,    3.07090647005439,  -18.5821,    76.0437],
    '1JHN':  [47.4798844844683,   10.92204561670947,   24.3222,    80.4187],
    '2JHN':  [3.1247536134185,    3.67345877025737,   -2.6209,    17.7436],
    '3JHN':  [0.9907298624944,    1.31538940138001,   -3.1724,    10.9712],
    '2JHH': [-10.2866051639817,    3.97960190019757,  -35.1761,    11.8542],
    '3JHH':  [4.7710233597359,    3.70498129755812,   -3.0205,    17.4841],
    }


def normalize_target(df, col='scalar_coupling_constant'):
    df = df.copy()    
    groups = df.groupby('type').groups
    for key, vals in COUPLING_TYPE_STATS.items():
        sub_type_df = df.loc[groups[key]]
        sub_type_df[col] -= vals[0]
        sub_type_df[col] /= vals[1]
        df.loc[groups[key], col] = sub_type_df[col]
    return df

def unnormalize_target(df, col='scalar_coupling_constant'):
    df = df.copy()    
    groups = df.groupby('type').groups
    for key, vals in COUPLING_TYPE_STATS.items():
        sub_type_df = df.loc[groups[key]]
        sub_type_df[col] *= vals[1]
        sub_type_df[col] += vals[0]
        df.loc[groups[key], col] = sub_type_df[col]
    return df

COUPLING_TYPE = ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']
COL_STATS = {
    }

def normalize_cols(df, cols=['scalar_coupling_constant']):
    global COL_STATS
    df = df.copy()        
    groups = df.groupby('type').groups
    
    for ctype in COUPLING_TYPE:
        sub_df = df.loc[groups[ctype]]
        for col in cols:
            key = f'{ctype}_{col}'            
            if key not in COL_STATS:
                COL_STATS[key] = [sub_df[col].mean(), sub_df[col].std(), sub_df[col].min(), sub_df[col].max()]
            
            sub_df[col] -= COL_STATS[key][0]
            sub_df[col] /= COL_STATS[key][1]
            df.loc[groups[ctype], col] = sub_df[col]
    return df


def unnormalize_cols(df, cols=['scalar_coupling_constant']):
    global COL_STATS
    df = df.copy()        
    groups = df.groupby('type').groups
    
    for ctype in COUPLING_TYPE:
        sub_df = df.loc[groups[ctype]]
        for col in cols:
            key = f'{ctype}_{col}'            
            sub_df[col] *= COL_STATS[key][1]
            sub_df[col] += COL_STATS[key][0]            
            df.loc[groups[ctype], col] = sub_df[col]
    return df


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def adjust_learning_rate(optimizer, epoch):  
    lr  = CFG.learning_rate 
    if epoch >= 40:
        lr = (CFG.lr_decay)**((epoch-20)//20) * CFG.learning_rate    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
    return lr

if __name__ == '__main__':
    main()
