from torch import nn, ones
from torch.autograd import Variable
from torchvision import models
from torch.nn.init import kaiming_normal
from torch import np
import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from p_data_augmentation import PowerPIL
import sys

from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

from torch import np, from_numpy  # Numpy like wrapper


class ImgTagsDualFeedDataset(Dataset):
    """Dataset wrapping images, labels and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
    """

    def __init__(self, csv_path, img_path, img_ext, vocab_mapping, transform=None):

        self.df = pd.read_csv(csv_path)
        assert self.df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
            "Some images referenced in the CSV file were not found"

        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X = self.df['image_name']

        self.vocab_mapping = vocab_mapping

        self.tags = self.df['tags'].str.split()

    def X(self):
        return self.X

    def __getitem__(self, index):

        img = Image.open(self.img_path + self.X[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        vocab = self.vocab_mapping
        tags = []
        tags.append(vocab['<BEGIN>'])
        tags.extend([vocab[tag] for tag in self.tags[index]])
        tags.append(vocab['<STOP>'])

        tags = torch.Tensor(tags)
        return img, tags

    def __len__(self):
        return len(self.df.index)

    def collate_fn(self, data):
        """Creates mini-batch tensors for tags with variable size

        Args:
            data: list of tuple (input, target).
                - image: torch tensor of shape (3, ?, ?).
                - target: torch tensor of same shape (?); variable length.
        Returns:
            images: torch tensor of shape (batch_size, 3, ?, ?).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded tags.
        """
        # Sort a data list by target length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)
        imgs, tags = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        imgs = torch.stack(imgs, 0)

        # Merge tags (from tuple of 1D tensor to 2D tensor).
        lengths = [len(tag) for tag in tags]
        targets = torch.zeros(len(tags), max(lengths)).long()
        for i, tag in enumerate(tags):
            end = lengths[i]
            targets[i, :end] = tag[:end]
        return imgs, targets, lengths


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



def main():
   normalize = transforms.Normalize(mean=[0.30249763, 0.34421128, 0.31507707],
                                         std=[0.13718571, 0.14363901, 0.16695975])
   ds_transform_augmented = transforms.Compose([
            transforms.RandomSizedCrop(224),
            PowerPIL(),
            transforms.ToTensor(),
            normalize,
   ])
   vocab = ['<BEGIN>', '<STOP>', 'clear', 'cloudy', 'haze', 'partly_cloudy',
                 'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
                 'blow_down', 'conventional_mine', 'cultivation', 'habitation',
                 'primary', 'road', 'selective_logging', 'slash_burn', 'water'
                 ]

   word_to_ix = {word: i for i, word in enumerate(vocab)}
   print(word_to_ix)
   one_hot_mapping = {k: np.eye(19)[v] for k, v in word_to_ix.items()}
   print(one_hot_mapping)

   X_train = ImgTagsDualFeedDataset('./data/train.csv', './data/train-jpg/', '.jpg',
                                    word_to_ix,
                                    ds_transform_augmented
                                    )
   train_loader = torch.utils.data.DataLoader(dataset=X_train,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True,
                                              collate_fn=X_train.collate_fn)
   print(X_train[1])
   model = CNN_RNN_Fused(19, 5, 2)
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
   print("start trining")

   epoch = 0
   for batch_idx, (img, tags, lengths) in enumerate(train_loader):
       img = Variable(img).cuda()
       tags = Variable(tags).cuda()
       targets = pack_padded_sequence(tags, lengths, batch_first=True)[0]

       model.zero_grad()

       # Predict one tag at a time
       outputs = model(img, tags, lengths)

       # check one tag
       print(targets)
       loss = criterion(outputs, targets)
       loss.backward()
       optimizer.step()

       if batch_idx % 100 == 0:
           print('Train Epoch: {:03d} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx , len(train_loader) ,
                      100. * batch_idx / len(train_loader), loss.data[0]))


if __name__ == '__main__':
    sys.exit(main())
