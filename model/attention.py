import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from model.utils import FCNet


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, embed_size):
        super(Attention, self).__init__()
        '''
        img_feature: [batch, embed_size]
        word_feature: [batch, qst_len, embed_size]
        alpha: [qst_len, 1]
        '''
        self.embed_size = embed_size
        self.v_proj = FCNet([v_dim, embed_size], dropout=0.3)
        self.q_proj = FCNet([q_dim, embed_size], dropout=0.3)
        self.nonlinear = FCNet([v_dim + q_dim, embed_size], dropout=0.3)
        self.linear = weight_norm(nn.Linear(embed_size, 1), dim=None)

    def forward(self, img_feature, word_feature):
        qst_len = word_feature.size(1)
        img_feature = self.v_proj(img_feature).repeat(1, qst_len, 1)
        word_feature = self.q_proj(word_feature)
        combined_feature = torch.cat([img_feature, word_feature], dim=2)

        alpha = self.nonlinear(combined_feature)  # squeeze(dim= 2) # [batch, qst_len, 1]
        alpha = self.linear(alpha)
        w = nn.functional.softmax(alpha, 1)
        return w
