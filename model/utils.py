import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class WordEmbedding(nn.Module):
    """Word Embedding
    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, embeddings, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False)
        assert self.emb.weight.shape == (ntoken, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class MRN(nn.Module):
    def __init__(self, qst_embed_size, img_embed_size, embed_size, dropout):
        super(MRN, self).__init__()
        self.qst_direct_fc = nn.Linear(qst_embed_size, embed_size)
        self.qst_fc = nn.Sequential(
            nn.Linear(qst_embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True)
        )
        self.img_fc = nn.Sequential(
            nn.Linear(img_embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True)
        )

    def forward(self, q, v):
        v_embed = self.img_fc(v)
        q_embed = self.qst_fc(q)
        vq = v_embed * q_embed
        q_direct = self.qst_direct_fc(q)
        h = vq + q_direct
        return h
