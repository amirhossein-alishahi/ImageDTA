import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import Sequential, Linear, ReLU
import torchvision.models as models

import transformerDecoder

from create_data import dileImage
from layer import LinkAttention

# ImageDTA  model
class ImageNet(torch.nn.Module):

    def __init__(self, num_features_xd=64, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(ImageNet, self).__init__()

        #drug
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,embed_dim))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, embed_dim))
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4, embed_dim))
        self.conv8 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8, embed_dim))
        self.conv10 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(10,embed_dim))
        self.conv12 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(12, embed_dim))
        self.conv16 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(16, embed_dim))
        self.conv32 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(32, embed_dim))
        self.conv64 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(64, embed_dim))

        self.embedding_xd = nn.Embedding(num_features_xd + 1, embed_dim)


        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters * 3, kernel_size=8)
        self.fc1_xt = nn.Linear(96 * 107, output_dim)

        #BiLSTM
        self.lstm = nn.LSTM(input_size=888,hidden_size=128,num_layers=2,dropout=0.3,batch_first=True,bidirectional=True)


        # FC layers
        self.fc1 = nn.Linear(1144, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles, protein):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Drug
        x = self.embedding_xd(smiles)
        x = torch.unsqueeze(x, 1)

        x1 = self.conv1(x)
        x1 = torch.squeeze(x1)
        x2 = self.conv2(x)
        x2 = torch.squeeze(x2)
        x4 = self.conv4(x)
        x4 = torch.squeeze(x4)
        x8 = self.conv8(x)
        x8 = torch.squeeze(x8)
        x10 = self.conv10(x)
        x10 = torch.squeeze(x10)
        x12 = self.conv12(x)
        x12 = torch.squeeze(x12)
        x16 = self.conv16(x)
        x16 = torch.squeeze(x16)
        x32 = self.conv32(x)
        x32 = torch.squeeze(x32)
        x64 = self.conv64(x)
        x64 = torch.squeeze(x64)
        #First Concatenation layer
        xd = torch.cat((x1, x2, x4, x8, x10, x12, x16, x32, x64), 1)

        # protein
        embedded_xt = self.embedding_xt(protein)

        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        xt = conv_xt.view(-1, 96 * 107)
        xt = self.fc1_xt(xt)

        #Second Concatenation layer
        xc = torch.cat((xd, xt), 1)

        #BiLSTM
        out, _ = self.lstm(xc)

        # Third Concatenation layer
        out = torch.cat((xd, xt, out), 1)
        # add some dense layers
        xc = self.fc1(out)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out