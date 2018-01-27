from torch import nn, ones
from torch.autograd import Variable
from torchvision import models
from torch.nn.init import kaiming_normal
from torch import np
import torch
import torch.nn.functional as F
import random
import numpy as np



class CNN_RNN_Fused(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_rnn_layers):
        super(CNN_RNN_Fused, self).__init__()

        ## CNN part
        # Loading ResNet arch from PyTorch and weights from Pycaffe
        original_model = models.resnet50(pretrained=False)
        #original_model.load_state_dict(torch.load('./zoo/resnet50.pth'))

        # Everything except the last linear layer
        self.convnet = nn.Sequential(*list(original_model.children())[:-1])

        # Get number of features of last layer
        num_feats_cnn = original_model.fc.in_features

        ## RNN part
        hidden_size = embed_dim  # for simplification
        self.vocab_size = vocab_size
        self.embeds = nn.Embedding(vocab_size,
                                   embed_dim)  # , padding_idx=0 Ignore the <start> (0 in vocab) for gradient
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_rnn_layers, batch_first=True)
        self.num_rnn_layers = num_rnn_layers

        ## Projection
        self.prj_cnn = nn.Linear(num_feats_cnn, embed_dim)
        self.prj_rnn = nn.Linear(hidden_size, embed_dim)

        ## Prediction
        # link embedding and decoding weight
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.fc.weight = self.embeds.weight

    def forward(self, img, tags, lengths, hidden=None):
        ## CNN
        cnn_feats = self.convnet(img)
        cnn_feats = cnn_feats.view(cnn_feats.size(0), -1)
        cnn_feats = self.prj_cnn(cnn_feats)

        tag_ids = []
        embed = self.embeds(tags)
        for _ in tags:
            ## RNN
            rnn_out, hidden = self.rnn(embed, hidden)

            ## Projection
            rnn_out = self.prj_rnn(rnn_out[:, 0, :])  # Extract the first prediction from sequence
            fuse = cnn_feats + rnn_out
            fuse = self.fc(fuse)
            predicted = fuse.max(1)[1]
            tag_ids.append(predicted)
            packed = self.embeds(predicted)
        tag_ids = torch.cat(tag_ids, 1)
        print(tag_ids)
        return tag_ids.squeeze()

    def genTags(self, inputs, states=None):
        tag_ids = []
        inputs = self.embeds(inputs)
        for i in range(self.vocab_size):  # maximum sampling length
            hiddens, states = self.rnn(inputs, states)  # (batch_size, 1, hidden_size)
            outputs = self.fc(hiddens.squeeze(1))  # (batch_size, vocab_size)
            # outputs = F.softmax(outputs)
            predicted = outputs.max(1)[1]
            tag_ids.append(predicted)
            inputs = self.embeds(predicted)
        tag_ids = torch.cat(tag_ids, 1)  # (batch_size, 19)
        return tag_ids.squeeze()