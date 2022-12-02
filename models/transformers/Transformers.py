import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from .positional_encoding import PositionalEncoding
from .decoder import Decoder
from .encoder import Encoder

from conf import *

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.dec_voc_size = dec_voc_size
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.is_train = True
        self.tar_len = 11
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        batch_size = src.shape[0]

        src_mask = self.make_pad_mask(src, src)

        src_trg_mask = self.make_pad_mask(trg, src)

        trg_mask = self.make_pad_mask(trg, trg) * \
                   self.make_no_peak_mask(trg, trg)

        enc_src = self.encoder(src, src_mask)
        
        output = torch.zeros(batch_size, self.tar_len, self.dec_voc_size).to(device)
        
        if self.is_train:
            decoder_input = trg
            for t in range(self.tar_len):
                trg = decoder_input[:,:(t+1)]

                src_trg_mask = self.make_pad_mask(trg, src)

                trg_mask = self.make_pad_mask(trg, trg) * \
                    self.make_no_peak_mask(trg, trg)
                decoder_out = self.decoder(trg, enc_src, trg_mask, src_trg_mask)

                output[:,t,:] = decoder_out[:,-1,:]
        else:
            for t in range(self.tar_len):

                src_trg_mask = self.make_pad_mask(trg, src)

                trg_mask = self.make_pad_mask(trg, trg) * \
                    self.make_no_peak_mask(trg, trg)
                decoder_out = self.decoder(trg, enc_src, trg_mask, src_trg_mask)

                predicted_ids = decoder_out[:,-1,:].argmax(dim=-1).unsqueeze(1)

                trg = torch.cat([trg, predicted_ids], dim=-1)

                output[:,t,:] = decoder_out[:,-1,:]

        return output

    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask