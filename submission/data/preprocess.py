from collections import Counter
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import download, data, pos_tag
from os.path import abspath, exists
NLTK_DIR = abspath("data/datasets/nltk/")
data.path.append(NLTK_DIR)

# Download NLTK data if not already existing
if not exists(abspath("data/datasets/nltk/corpora/stopwords/english")):
    download("stopwords",
             download_dir=NLTK_DIR)

if not exists(abspath("data/datasets/nltk/corpora/wordnet")):
    download("wordnet",
             download_dir=NLTK_DIR)

if not exists(abspath("data/datasets/nltk/tokenizers/punkt")):
    download("punkt",
             download_dir=NLTK_DIR)

if not exists(abspath("data/datasets/nltk/taggers/averaged_perceptron_tagger")):
    download("averaged_perceptron_tagger",
             download_dir=NLTK_DIR)


STOPWORDS = set(stopwords.words('english'))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def count_numbers(text):
    count=0
    tokens = text.split(" ")
    for t in tokens:
        if is_number(t):
            count +=1
    
    return count

def label_selective_tagging(text):
    '''
    Remove any numbers that are not labelled as a subject
    '''

    if count_numbers(text) > 2:
        sentence = text.split(' ')
        labels = Counter(sentence).items()
        labels = sorted(labels, key=lambda item: item[1], reverse=True)
        label_candidates = []
        for lbl, freq in labels:
            if freq > 1 and not lbl in STOPWORDS:
                label_candidates.append(lbl)

        window = 4
        output = []
        for j, word in enumerate(sentence):
            try:
                # word is a number
                n = float(word)
                lookahead = sentence[j + 1:min(j + window + 1, len(sentence))]
                lookbehind = sentence[max(j - window, 0):min(j, len(sentence))]

                if lookahead[0] in ["are", "have", "were", "is", "with", "of"]:
                    output.append(word)
                    continue

                if lookahead[0] != label_candidates[0]:
                    continue

                if lookbehind[-1] in ["has", "holds", "contains", "$"]:
                    output.append(word)
                    continue

                if lookbehind[-1] in ["and"] and not lookahead[0] == label_candidates[0]:
                    continue

                move_on = False
                for lbl in lookahead:
                    if '.' == lbl:
                        break

                    if lbl == label_candidates[0]:
                        output.append(word)
                        move_on = True
                        break

                if move_on:
                    continue

                for lbl in lookbehind:
                    if lbl in label_candidates[:1]:
                        output.append(word)
                        break
            except:
                output.append(word)

        o = ' '.join(output)

        if count_numbers(o) >= 2:
            return o
        else:
            return text
    else:
        return text
