import torch
from data.preprocess import label_selective_tagging
from data.NumberTag import NumberTag
from utils import *
from conf import *
from convert import *

def idx_to_word(x, loader):
    words = []
    for i in x:
        word = loader.idx_to_word(i.item(),False)

        if word not in ["<pad>", "<unk>", "<sos>", "<eos>"]:
            words.append(word)
    words = " ".join(words)
    return words

def preprocess(sent, loader):
    
    sent = label_selective_tagging(sent)
    tagger = NumberTag(sent, "x = 0 ")
    masked_text, _, lookup = tagger.get_masked()
    
    if tagger.mapped_correctly():
        sent = loader.tokenizer(masked_text)
    print("replaced with tag : ", sent)
    input = loader.process([sent],True).reshape(1,-1)

    return input, lookup



def translate(inp_sent, loader, model=None):
    # model.eval()
    print(" input given : ",inp_sent)
    # inp is currently in the form of sentence or string of words separated by spaces
    inp_tensor, lookup = preprocess(inp_sent, loader)

    # print(inp_tensor.shape)
    inp_tensor = inp_tensor.to(device)

    sos_token = loader.target_vocab["<sos>"]
    eos_token = loader.target_vocab["<eos>"]
    
    target = torch.tensor([sos_token])

    output = target.unsqueeze(0).to(device)

    # print(output.shape)

    model.eval()
    with torch.no_grad():
        output = model(inp_tensor, output)

    output = output.argmax(dim=-1).squeeze()
    print(output.shape, output)
    
    equation = idx_to_word(output, loader)
    tokenized_eq = equation.split(" ")

    for i in range(len(tokenized_eq)):
        if "<" in tokenized_eq[i]:
            try:
                tokenized_eq[i] = lookup[tokenized_eq[i]]
            except:
                pass
    
    print("question : ", inp_sent)
    print("postfix equation : ", equation)
    print("postfix equation replaced : ", " ".join(tokenized_eq))
    try:
        print("infix equation : ", getInfix(" ".join(tokenized_eq)))
        print(evaluatePostfix())
    except:
        pass


if __name__=="__main__":
    loader = load_data_from_binary("./dataloaders/loader-postfix.pkl")
    I = "sally had 27 pokemon cards . dan gave her 41 new pokemon cards . sally bought 20 pokemon cards . How many pokemon cards does sally have now ? "
    translate(I, loader)
