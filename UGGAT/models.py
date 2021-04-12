import torch
import torch.nn as nn
import torch.nn.functional as F
from UGGAT.layers import GraphAttentionLayer, SpGraphAttentionLayer,UG_GraphAttentionLayer



class UGGAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(UGGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [UG_GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att1 = GraphAttentionLayer(nhid * nheads, 256, dropout=dropout, alpha=alpha, concat=False)
        self.out_att2 = GraphAttentionLayer(256, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, uncertainty):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, uncertainty) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att2(x, adj))
        x = x[0,:].unsqueeze(0)
        return x
