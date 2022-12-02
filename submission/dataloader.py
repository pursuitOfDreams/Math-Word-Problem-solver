from dataset import MWPDataset
from conf import *
from torch.utils.data import DataLoader
from collections import OrderedDict, Counter
import torch
from torchtext.vocab import vocab
from itertools import chain
from tqdm import tqdm

class MWPDataLoader:

    def __init__(self, tokenizer, init_token, eos_token):
        self.tokenizer = tokenizer
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.is_vocab = False
        self.fix_length_source = 60
        self.fix_length_tar = 10

    def make_dataset(self):
        self.train_dataset = MWPDataset(path= TRAIN_DATA_PATH, tokenizer=self.tokenizer)
        self.test_dataset = MWPDataset(path= TEST_DATA_PATH, tokenizer=self.tokenizer)


    def build_vocab(self):
        specials = list(OrderedDict.fromkeys(
                        tok for tok in [self.unk_token, self.pad_token, self.init_token,
                                        self.eos_token] if tok is not None))

        words = set()
        
        for data in tqdm(self.train_dataset.texts, desc = "train text vocab"):
            for x in data:
                words.add(x)
        
        words.add("<sos>")
        words.add("<eos>")
        words.add("<pad>")
        words.add("<unk>")

        self.source_vocab = {word: i for i, word in enumerate(words)}
        self.source_vocab_rev = { i: word for i, word in enumerate(words)}

        words = set()
        for data in tqdm(self.train_dataset.equations, desc = "train eq vocab"):
            for x in data:
                words.add(x)

        words.add("<sos>")
        words.add("<eos>")
        words.add("<pad>")
        words.add("<unk>")

        self.target_vocab = {word: i for i, word in enumerate(words)}
        self.target_vocab_rev = {i:word for i, word in enumerate(words)}
        self.is_vocab=True

    def word_to_idx(self, word, source=True):
        if source:
            return self.source_vocab.get(word, self.source_vocab["<unk>"])
        else:
            return self.target_vocab.get(word, self.target_vocab["<unk>"])

    def idx_to_word(self, idx, source=True):
        if source:
            return self.source_vocab_rev.get(idx, "<unk>")
        else:
            return self.target_vocab_rev.get(idx, "<unk>")


    def pad(self, data, is_source=True):
        max_len = None
        if is_source:
            max_len = self.fix_length_source+2
        else:
            max_len = self.fix_length_tar+2
            
        padded = []
        for x in data:
            padded.append(
                [self.init_token]+
                list(x[:max_len])+
                [self.eos_token]+
                [self.pad_token]*max(0, max_len-len(x))
            )

        return padded

    def numericalize(self, arr, is_source = True, device= None):

        if self.is_vocab==False:
            raise ValueError(" No vocab to use right now ")
        if is_source:
            arr = [[self.word_to_idx(x, is_source) for x in ex] for ex in arr]
        else:
            arr = [[self.word_to_idx(x, is_source) for x in ex] for ex in arr]

        var = torch.tensor(arr, device= device)

        return var

    def process(self, data, is_source):
        padded = self.pad(data, is_source)
        # print(padded[0])
        tensor = self.numericalize(padded, is_source)
        return tensor
        
    def apply_processing(self):
        self.train_dataset.texts = self.process(self.train_dataset.texts,True)
        self.train_dataset.equations = self.process(self.train_dataset.equations, False)
        self.test_dataset.texts = self.process(self.test_dataset.texts, True)
        self.test_dataset.equations = self.process(self.test_dataset.equations, False)
    

    def make_iter(self):
        train_iter = DataLoader(
                              self.train_dataset, 
                              batch_size=train_batch_size,
                              shuffle=True,
                            )


        test_iter = DataLoader(
                                self.test_dataset, 
                                batch_size = test_batch_size, 
                                shuffle = True
                            )

        return train_iter, test_iter

