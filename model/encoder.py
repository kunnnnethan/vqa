import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

from model.utils import WordEmbedding


class ImgEncoder(nn.Module):
    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
        model = models.resnet101(pretrained=True)
        in_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-1])

        self.model = model  # loaded model without last fc layer
        self.fc = nn.Linear(in_features, embed_size)  # feature vector of image

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            img_feature = self.model(image)  # [batch_size, vgg16(19)_fc=4096]
        img_feature = torch.squeeze(img_feature, dim=2)
        img_feature = torch.squeeze(img_feature, dim=2)
        img_feature = self.fc(img_feature)  # [batch_size, embed_size]
        l2_norm = nn.functional.normalize(img_feature, p=2, dim=1)  # l2-normalized feature vector

        return l2_norm


class QstEncoder(nn.Module):
    def __init__(self, word_embed_size, embed_size, num_layers, hidden_size, word_embeddings, embeddings_size):
        super(QstEncoder, self).__init__()
        self.word_embedding = WordEmbedding(word_embeddings, embeddings_size, word_embed_size, 0.0)
        self.gru = nn.GRU(word_embed_size, hidden_size, num_layers, batch_first=False)
        #self.layer_norm = nn.LayerNorm((num_layers * hidden_size,))
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (self.num_layers, batch, self.hidden_size)
        return Variable(weight.new(*hid_shape).zero_())

    def forward(self, question):
        batch = question.size(0)
        qst_vec = self.word_embedding(question)
        qst_vec = qst_vec.transpose(0, 1)

        hidden = self.init_hidden(batch)
        word_feature, qst_feature = self.gru(qst_vec, hidden)  # (batch_size, sqe_len, num_directions * hidden_size)
        qst_feature = qst_feature.transpose(0, 1)  # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        #qst_feature = self.layer_norm(qst_feature)
        return qst_feature
