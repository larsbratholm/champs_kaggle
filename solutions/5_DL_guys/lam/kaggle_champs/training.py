import torch
from tqdm.autonotebook import tqdm
from .metrics import AverageMetric, MeanLogGroupMAE


def train_epoch(global_iteration, epoch, model, device, optimizer, train_loader, tb_logger, gradient_accumulation_steps=1):
    model.train()

    avg_loss = AverageMetric()
    log_mae = MeanLogGroupMAE()
    
    pbar = tqdm(train_loader)
    for step, data in enumerate(pbar):
        data = data.to(device)
        
        pred = model(data)

        loss = torch.nn.L1Loss(reduction='none')(pred.view(-1),
                                                 data.y.view(-1))
        loss = (loss *
                data.sample_weight.view(-1)).sum() / data.sample_weight.sum()

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            global_iteration += 1

        tb_logger.add_scalar('loss', loss.item(), global_iteration)

        avg_loss.update(loss.item() * data.num_graphs, data.num_graphs)
        log_mae.update(pred.view(-1), data.y.view(-1), data.type)

        pbar.set_postfix_str(f'loss: {avg_loss.compute():.4f}')
    return avg_loss.compute(), log_mae, global_iteration