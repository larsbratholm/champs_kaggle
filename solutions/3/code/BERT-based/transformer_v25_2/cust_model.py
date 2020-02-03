import torch
import db
from torch import nn as nn
from pytorch_transformers.modeling_bert import (BertConfig, BertEncoder, BertPooler, BertLayerNorm)

class SelfAttn(nn.Module):
    def __init__(self, config):
        super(SelfAttn, self).__init__()
        self.config = config
        self.hsize = 64
        self.atom_emb = nn.Embedding(5, 64)
        self.type_emb = nn.Embedding(15, 64)
        self.pos_emb = nn.Linear(3, 256, bias=False)
        self.dist_emb = nn.Linear(1, 64, bias=False)
        self.mu_emb = nn.Linear(1, 32, bias=False) # dipole_moment

        self.attn = BertEncoder(config)
        def get_reg_layer(output_size):
            return  nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),            
                nn.LeakyReLU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),            
                nn.LeakyReLU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, output_size),
            )
        self.reg_layers4 = nn.ModuleList([get_reg_layer(4) for _ in range(9)])
        self.reg_layers1 = nn.ModuleList([get_reg_layer(1) for _ in range(9)])
        # not currently used.
        self.reg_aux = None

        #self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, atom0, atom1, typ, xyz0, xyz1, mu0, mu1, mask):
        atom0_emb = self.atom_emb(atom0)
        atom1_emb = self.atom_emb(atom1)
        type_emb = self.type_emb(typ)
        xyz0_emb = self.pos_emb(xyz0)
        xyz1_emb = self.pos_emb(xyz1)
        dist_emb = self.dist_emb((xyz0-xyz1).norm(dim=2, keepdim=True))
        mu0_emb = self.mu_emb(mu0)
        mu1_emb = self.mu_emb(mu1)


        atom_pairs = torch.cat([atom0_emb, atom1_emb, type_emb, 
                                xyz0_emb, xyz1_emb, dist_emb, mu0_emb, mu1_emb], 2)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers

        encoded_layers = self.attn(atom_pairs, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        batch_size = typ.size(0)
        typ = typ.view(-1)
        sequence_output = sequence_output.view(-1, sequence_output.size(-1))

        org_indices = []
        type_output1 = []
        type_output4 = []
        for i in range(15):
            typ_bool = (typ == i)
            if typ_bool.sum() == 0: continue
            org_indices.append(typ_bool.nonzero().view(-1))

            if i > 8:
                i = 8
            type_output4.append( self.reg_layers4[i](sequence_output[typ_bool] ) )        
            type_output1.append( self.reg_layers1[i](sequence_output[typ_bool] ) )
        org_indices = torch.cat(org_indices)
        _, rev_indices = org_indices.sort()
        type_output1 = torch.cat(type_output1)
        type_output4 = torch.cat(type_output4)
        type_output1 = type_output1[rev_indices]
        type_output4 = type_output4[rev_indices]

        #res_indices = torch.cuda.LongTensor(org_indices)[rev_indices]
        pred4 = type_output4.view(batch_size, -1, type_output4.size(-1))
        pred1 = type_output1.view(batch_size, -1, type_output1.size(-1))
        pred5 = torch.cat([pred1, pred4], -1)
        return pred1, pred5





