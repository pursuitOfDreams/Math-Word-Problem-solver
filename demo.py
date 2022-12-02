import torch
from models.transformers.Transformers import Transformer
from torch import nn
from conf import *
from dataloader import *
from predict import *

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)


model.load_state_dict(torch.load("./best_models/postfix/model-postfix1.pt"))
model.eval()
model.is_train = False

while True:
    inp_sent = input("Give an input ")
    inp_sent = inp_sent.lower()

    translate(inp_sent, loader, model)

    print()
    
