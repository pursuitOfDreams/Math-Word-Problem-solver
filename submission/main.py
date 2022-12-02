# from models.transformers.transformers import TransformerModel
# from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from time import time
from random import seed, shuffle
from os import makedirs
from os.path import exists
import utils
from data.NumberTag import NumberTag
import torch
from collections import Counter
from tqdm import tqdm
from data.preprocess import label_selective_tagging
import math
from models.transformers.trainer import train
from models.transformers.Transformers import Transformer
import time
from torch import nn
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from conf import *
import re
from dataloader import MWPDataLoader

SEED = 0
DATA_PATH = "data/train/train_all_prefix.pkl"
CHK_POINT_FOLDER = "models/transformers/checkpoint/"


# NUM_LAYERS = 
# D_MODEL = 
# DFF = 
# NUM_HEADS = 
# DROPOUT = 
BATCH_SIZE = 128
# EPOCHS = 

torch.manual_seed(SEED)
seed(SEED)

tokenizer_text = None
tokenizer_eq = None

SOS_txt = None
EOS_txt = None
SOS_eq = None
EOS_eq = None

trained_model_path = "./models/trained/"

tokenizer = get_tokenizer("basic_english")

loader = MWPDataLoader(
                        tokenizer=tokenizer,
                        init_token="<sos>",
                        eos_token="<eos>"
                    )

loader.make_dataset()
loader.build_vocab()
loader.apply_processing()
train_iter, test_iter = loader.make_iter()

src_pad_idx = loader.source_vocab['<pad>']
trg_pad_idx = loader.target_vocab['<pad>']
trg_sos_idx = loader.target_vocab['<sos>']


enc_voc_size = len(loader.source_vocab)
dec_voc_size = len(loader.target_vocab)

print(enc_voc_size)
print(dec_voc_size)

print(trg_sos_idx, loader.target_vocab["<eos>"], trg_pad_idx, loader.target_vocab["<unk>"])

def idx_to_word(x):
    words = []
    for i in x:
        word = loader.idx_to_word(i.item(),False)

        if word not in ["<pad>", "<unk>", "<sos>", "<eos>"]:
            words.append(word)
    words = " ".join(words)
    return words



def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

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

optimizer = torch.optim.Adam(model.parameters(),lr = lr)

criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)


# def bleu_stats(hypothesis, reference):
#     """Compute statistics for BLEU."""
#     stats = []
#     stats.append(len(hypothesis))
#     stats.append(len(reference))
#     for n in range(1, 5):
#         s_ngrams = Counter(
#             [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
#         )
#         r_ngrams = Counter(
#             [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
#         )

#         stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
#         stats.append(max([len(hypothesis) + 1 - n, 0]))
#     return stats

# def bleu(stats):
#     """Compute BLEU given n-gram statistics."""
#     if len(list(filter(lambda x: x == 0, stats))) > 0:
#         return 0
#     (c, r) = stats[:2]
#     log_bleu_prec = sum(
#         [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
#     ) / 4.
#     return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

# def get_bleu(hypotheses, reference):
#     """Get validation BLEU score for dev set."""
#     stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
#     for hyp, ref in zip(hypotheses, reference):
#         stats += np.array(bleu_stats(hyp, ref))
#     return 100 * bleu(stats)

def get_precision(r,h):
        # Uses a bias toward the real translation to
        #  determine a similarity score between space
        #  separated strings.
        r = r.split()
        h = h.split()

        comparison_list = []

        for i, char in enumerate(h):
            if i < len(r):
                if r[i] == char:
                    comparison_list.append(1)
                else:
                    comparison_list.append(0)
            else:
                comparison_list.append(0)

        precision = sum(comparison_list) / len(comparison_list)

        short_precision = "%1.4f" % precision
        return float(short_precision)

def get_bleu(hypothesis_list):
    bleu_avg = []
    perfect = []
    precision =[]

    for hypothesis, actual in hypothesis_list:
        hypothesis = re.sub(r".0", "", hypothesis)
        actual = re.sub(r".0", "", actual)

        print(actual, hypothesis)
        pc = get_precision(actual, hypothesis)
        precision.append(pc)

        bleu_hyp = hypothesis.split()
        bleu_act = actual.split()

        min_length = min(len(bleu_act), len(bleu_hyp))

        score = "%1.4f" % sentence_bleu([bleu_act],
                                            bleu_hyp,
                                            weights=(0.5, 0.5),
                                            smoothing_function=SmoothingFunction().method2)

        if score[0] == '1':
                perfect.append((hypothesis, actual))

        bleu_avg.append(float(score))

    number_perfect = len(perfect)

    number_of_attempts = len(bleu_avg)

    perfection_percentage = (number_perfect / number_of_attempts) * 100

    short_percentage = float("%3.2f" % perfection_percentage)

    avg_precision = (sum(precision)/len(precision)) * 100

    short_precision = float("%3.2f" % avg_precision)

    bleu = sum(bleu_avg)/len(bleu_avg) * 100

    short_bleu = float("%3.2f" % (bleu))

    return number_of_attempts, short_percentage, short_precision, short_bleu



def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        total_bleu = []
        count = 0
        for i, batch in tqdm(enumerate(iterator), desc="batches"):
            src, trg = batch
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            
            x, y = batch
            y = y[:, 1:]

            for j in range(len(y)):
                trg_words = idx_to_word(y[j])
                output_words = output[j].argmax(dim=1)
                output_words = idx_to_word(output_words)
                if count < 50:
                    print(trg_words, output_words,sep="\t")
                count+=1
                # bleu = sentence_bleu(hypothesis=output_words.split(), references=trg_words.split())
                total_bleu.append((output_words, trg_words))
                
        print(len(total_bleu))
        number_of_attempts, short_percentage, short_precision, short_bleu = get_bleu(total_bleu) 

    # batch_bleu = sum(batch_bleu) / len(batch_bleu)
    # return epoch_loss / len(iterator)
    return epoch_loss / len(iterator), short_bleu

def run(total_epoch, best_loss):
    if not exists("./result/"):
        makedirs("./result/")
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip, device)
        # valid_loss, bleu = evaluate(model, test_iter, criterion)
        valid_loss = evaluate(model, test_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        # bleus.append(bleu)

        elapsed_time = end_time - start_time
        epoch_mins = int(elapsed_time / 60)
        epoch_secs = int(elapsed_time - (epoch_mins * 60))


        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(step))

        

        f = open('result/train_loss.txt', 'a')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'a')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'a')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.5f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.5f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        # print(f'\tBLEU Score: {bleu:.7f}')

if __name__=="__main__":
    start = time.time()

    if not exists(trained_model_path):
        makedirs(trained_model_path)

    if not exists(CHK_POINT_FOLDER):
        makedirs(CHK_POINT_FOLDER)
    
    
        
    # run(total_epoch=epochs, best_loss=inf)

    _, bleu= eva
    end = time.time()
    print(f"time taken : {end-start}")




    


    