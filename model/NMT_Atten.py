# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class NMT_Atten(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, source_embed_dim, target_embed_dim, source_embeddings,\
                            target_embeddings, hidden_size, target_max_len, bos_id):
        super(NMT_Atten, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.max_len = target_max_len
        self.bos_id = bos_id
        self.encoder = EncoderRNN(source_vocab_size, source_embed_dim, source_embeddings, hidden_size)
        # self.decoder = DecoderRNN(target_vocab_size, target_embed_dim, target_embeddings, hidden_size)
        self.decoder = AttentionDecoderRNN(target_vocab_size, target_embed_dim, target_embeddings, hidden_size)

    def forward(self, source_wordid, target_wordid):
        # source_wordid: batch_size, seq_len
        # target_wordid: batch_size, seq_len (include <bos> and <eos>)
        # hn/cn: num_layers * num_directions, batch_size, hidden_size
        output, hn, cn = self.encoder(source_wordid)
        hn = torch.cat((hn[-2], hn[-1]), -1)
        cn = torch.cat((cn[-2], cn[-1]), -1)
        # print(hn.size(), cn.size())
        # hn = torch.sum(hn, 0)
        # cn = torch.sum(cn, 0)
        target_wordid = target_wordid.permute(1, 0)
        target_wordid = target_wordid[:-1]
        # pred: seq_len, batch_size, target_vocab_size
        pred = torch.zeros((target_wordid.size(0), target_wordid.size(1), self.target_vocab_size)).float()
        if torch.cuda.is_available():
            pred = pred.cuda()
        for ii, nextword_id in enumerate(target_wordid):
            linear_out, hn, cn = self.decoder(hn, cn, nextword_id, output)
            pred[ii] = linear_out

        return pred.permute(1, 0, 2)

    def translate(self, source_wordid):
        # source_wordid: batch_size, seq_len
        output, hn, cn = self.encoder(source_wordid)
        hn = torch.cat((hn[-2], hn[-1]), -1)
        cn = torch.cat((cn[-2], cn[-1]), -1)
        # hn = torch.sum(hn, 0)
        # cn = torch.sum(cn, 0)
        nextword_id = torch.LongTensor([self.bos_id]).expand(source_wordid.size(0))
        pred_word = torch.zeros((self.max_len, source_wordid.size(0))).long()
        if torch.cuda.is_available():
            nextword_id = nextword_id.cuda()
            pred_word = pred_word.cuda()
        for ii in range(self.max_len):
            linear_out, hn, cn = self.decoder(hn, cn, nextword_id, output)
            # pred: batch_size
            _, pred = torch.max(linear_out, 1)
            nextword_id = pred
            pred_word[ii] = pred

        return pred_word.permute(1, 0)


# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, source_vocab_size, emb_dim, embedding_weights, hidden_size, num_layer=1, finetuning=True):
        super(EncoderRNN, self).__init__()
        self.word_embed = nn.Embedding(source_vocab_size, emb_dim)
        if isinstance(embedding_weights, torch.Tensor):
            self.word_embed.weight = nn.Parameter(embedding_weights, requires_grad=finetuning)
        self.lstm = nn.LSTM(emb_dim, hidden_size//2, num_layers=num_layer, bidirectional=True)

    def forward(self, wordid_input):
        # wordid_input: batch_size, seq_len
        input_ = self.word_embed(wordid_input) # input_: batch_size, seq_len, emb_dim
        input_ = input_.permute(1, 0, 2) # input_: seq_len, batch_size, emb_dim
        output, (hn, cn) = self.lstm(input_)
        output = output.permute(1, 0, 2)  # output: batch_size, seq_len, hidden_size
        return output, hn, cn


# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, target_vocab_size, emb_dim, embedding_weights, hidden_size, finetuning=True):
        super(DecoderRNN, self).__init__()
        self.word_embed = nn.Embedding(target_vocab_size, emb_dim)
        if isinstance(embedding_weights, torch.Tensor):
            self.word_embed.weight = nn.Parameter(embedding_weights, requires_grad=finetuning)
        self.lstmcell = nn.LSTMCell(emb_dim, hidden_size)
        self.hidden2word = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, hidden, cell, nextword_id, encoder_output=None):
        # hidden/cell: batch_size, hidden_size
        # encoder_hidden = torch.sum(encoder_hidden, 0)
        # encoder_cell = torch.sum(encoder_cell, 0)
        input_ = self.word_embed(nextword_id) # batch_size, emb_dim
        # print(input_.size())
        # print(hidden.size())
        # print(cell.size())
        h1, c1 = self.lstmcell(input_, (hidden, cell))
        output = self.hidden2word(h1)

        return output, h1, c1


# Attention Decoder
class AttentionDecoderRNN(nn.Module):
    def __init__(self, target_vocab_size, emb_dim, embedding_weights, hidden_size, finetuning=True):
        super(AttentionDecoderRNN, self).__init__()

        self.word_embed = nn.Embedding(target_vocab_size, emb_dim)
        if isinstance(embedding_weights, torch.Tensor):
            self.word_embed.weight = nn.Parameter(embedding_weights, requires_grad=finetuning)
        self.lstmcell = nn.LSTMCell(emb_dim+hidden_size, hidden_size)
        # self.attention = nn.Linear(2*hidden_size, 1)
        v_size = int(0.5*hidden_size)
        self.w1 = nn.Linear(hidden_size, v_size, bias=False)
        self.w2 = nn.Linear(hidden_size, v_size, bias=False)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(v_size, 1, bias=False)
        self.hidden2word = nn.Linear(hidden_size, target_vocab_size)

    def attention(self, hidden, encoder_output):
        atten1 = self.w1(hidden).unsqueeze(1) # batch_size, 1, v_size
        atten2 = self.w2(encoder_output) # batch_size, seq_len, v_size
        atten3 = self.tanh(atten1+atten2) # batch_size, seq_len, v_size
        atten = self.v(atten3).squeeze(-1) # batch_size, seq_len
        atten_weight = F.softmax(atten, -1).unsqueeze(1) # batch_size, 1, seq_len
        atten_encoder_hidden = torch.bmm(atten_weight, encoder_output).squeeze(1) # batch_size, hidden_size

        return atten_encoder_hidden

    def forward(self, hidden, cell, nextword_id, encoder_output):
        # hidden/cell: batch_size, hidden_size
        # encoder_output: batch_size, seq_len, hidden_size
        input_ = self.word_embed(nextword_id) # batch_size, emb_dim
        # batch_size, seq_len, hidden_size = encoder_output.size()
        # Q = hidden.unsqueeze(1).expand(batch_size, seq_len, hidden_size)
        # X = torch.cat((encoder_output, Q), 2)
        # atten = self.attention(X).squeeze(-1) # batch_size, seq_len
        # atten_weight = F.softmax(atten, -1).unsqueeze(1) # batch_size, 1, seq_len
        # atten_encoder_hidden = torch.bmm(atten_weight, encoder_output).squeeze(1) # batch_size, hidden_size
        atten_encoder_hidden = self.attention( hidden, encoder_output)
        input_ = torch.cat((input_, atten_encoder_hidden), 1)
        h1, c1 = self.lstmcell(input_, (hidden, cell))
        output = self.hidden2word(h1)
        
        return output, h1, c1
