from service import PytorchDKT
from args import parse_args

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


args = parse_args(mode='train')

assessment_class = np.load(os.path.join(args.asset_dir, 'assessmentItemID_classes.npy'))
test_class = np.load(os.path.join(args.asset_dir, 'testId_classes.npy'))
tag_class = np.load(os.path.join(args.asset_dir, 'KnowledgeTag_classes.npy'))

args.n_questions = len(assessment_class)
args.n_test = len(test_class)
args.n_tag = len(tag_class)

model_path = os.path.join(args.model_dir, f'{args.config}.pt')
load_state = torch.load(model_path)
model = LSTM(args)
model.load_state_dict(load_state['state_dict'], strict=True)
   
test = pd.read_csv("questions.csv")

bento_dkt = PytorchDKT()
bento_dkt.pack('model', model)
bento_dkt.pack('test', test)
bento_dkt.pack('config', args)
bento_dkt.pack('assessmentItemID_classes', assessment_class)
bento_dkt.pack('testId_classes', test_class)
bento_dkt.pack('KnowledgeTag_classes', tag_class)

saved_path = bento_dkt.save()
print(saved_path)
