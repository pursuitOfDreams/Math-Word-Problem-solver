import copy 
import time
from torch import nn
import torch
import math


def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src, trg = batch
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg[:,:-1])
        output_reshape= output.contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()

        # gradient clipping in order to avoid gradient explosion and gradient decay
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss/ len(iterator)




    

