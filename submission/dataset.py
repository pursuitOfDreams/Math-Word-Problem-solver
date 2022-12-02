import torchtext
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from data.NumberTag import NumberTag
import utils
from data.preprocess import label_selective_tagging


class MWPDataset(Dataset):

    def __init__(self, path, tokenizer):
        self.texts = []
        self.equations = []
        self.tokenizer = tokenizer

        examples = utils.load_data_from_binary(path)
        shuffle(examples)


        print("processing examples")
        for example in tqdm(examples):
            txt, exp = utils.get_example_as_tuple(example)
            exp = utils.expressionize(exp)
            txt = label_selective_tagging(txt)

            tagger = NumberTag(txt, exp)
            masked_text, masked_exp, _ = tagger.get_masked()

            # function to check if all the numbers are correctly mapped with variables or not
            if tagger.mapped_correctly():
                self.texts.append(self.tokenizer(masked_text))
                self.equations.append(self.tokenizer(masked_exp))

    def __len__(self):
        return len(self.equations)

    
    def __getitem__(self, item):
        return self.texts[item], self.equations[item]


