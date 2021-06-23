import torch.nn as nn
from model.encoder import ImgEncoder, QstEncoder
from model.utils import MRN
from model.classifier import SimpleClassifier


class VqaModel(nn.Module):

    def __init__(self, embed_size, ans_vocab_size, word_embed_size, num_layers, hidden_size, word_embeddings, embeddings_size):
        super(VqaModel, self).__init__()
        # encoder
        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(word_embed_size, embed_size, num_layers, hidden_size, word_embeddings, embeddings_size)

        # MRN
        self.MRN1 = MRN(num_layers * hidden_size, embed_size, embed_size, 0.2)  # (qst_embed, img_embed, embed_size)
        self.MRN2 = MRN(embed_size, embed_size, embed_size, 0.2)

        # classifier
        self.classifier = SimpleClassifier(embed_size, ans_vocab_size, 0.5)

    def forward(self, img, qst):
        img_feature = self.img_encoder(img)  # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)

        h1 = self.MRN1(qst_feature, img_feature)
        h2 = self.MRN2(h1, img_feature)
        output = self.classifier(h2)

        return output
