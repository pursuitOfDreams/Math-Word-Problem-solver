import pickle
import re

def load_data_from_binary(absolute_path):
    with open(absolute_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_example_as_tuple(example):
    data = dict(example)
    return data["question"], data["equation"]

def save_data_to_binary(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def expressionize(data):
    data = re.sub(r"([a-z] \=|\= [a-z])", "", data)
    return data