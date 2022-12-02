import os
import sys
import yaml
import json
import pickle
import re
import random
from equation_converter import EquationConverter
from word2number import w2n

test_split = 0.05
# dataset specific

random.seed(0)
ai2 = []
illinois = []
commoncore = []
mawps =[]
name_to_dataset = {}

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

def save_data_to_binary(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def clean_sentence(text):
    # Clean up the data and separate everything by spaces
    text = re.sub(r"(?<!Mr|Mr|Dr|Ms)(?<!Mrs)(?<![0-9])(\s+)?\.(\s+)?", " . ",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"(\s+)?\?(\s+)?", " ? ", text)
    text = re.sub(r",", "", text)
    text = re.sub(r"^\s+", "", text)
    text = text.replace('\n', ' ')
    text = text.replace("'", " '")
    text = text.replace('%', ' percent')
    text = text.replace('$', ' $ ')
    text = re.sub(r"\.\s+", " . ", text)
    text = re.sub(r"\s+", ' ', text)

    sent = []
    for word in text.split(' '):
        try:
            sent.append(str(w2n.word_to_num(word)))
        except:
            sent.append(word)

    return ' '.join(sent)

def remove_variables(lst):
    new_lst = []

# print(DIR_PATH)

def transform_dataset(dataset_name):
    path = os.path.join(DIR_PATH,f"./datasets/{dataset_name}/questions.json")
    problem_list =[]

    with open(path, encoding='utf-8-sig') as f:
        json_data = json.load(f)

    for idx in range(len(json_data)):
        problem = []
        flag = True

        data = json_data[idx]
        if "sQuestion" in data and "lEquations" in data and "lSolutions" in data:
            for key, value in data.items():
                if key == "sQuestion" or key == "lEquations" or key == "lSolutions":
                    if len(value) == 0 or (len(value) > 1 and (key == "lEquations" or key == "lSolutions")):
                            flag = False
                    if key == "sQuestion":
                        desired_key = "question"
                        value = clean_sentence(value)
                        problem.append((desired_key,value.lower()))
                    
                    elif key == "lEquations":
                        # questions with multiple equations have been ignored
                        if dataset_name =="MaWPS" and len(value) > 1:
                            continue
                        
                        desired_key = "equation"
                        value = value[0]
                        problem.append((desired_key, value.lower()))
                    elif key=="lSolutions":
                        desired_key = "answer"
                        problem.append((desired_key, value[0]))
                    else:
                        problem.append((desired_key, value.lower()))

        if flag and len(problem)>0:
            problem_list.append(problem)
            if dataset_name=="MaWPS":
                mawps.append(problem)
            elif dataset_name=="CommonCore":
                commoncore.append(problem)
            elif dataset_name=="Illinois":
                illinois.append(problem)
            else:
                raise NotImplementedError(f"Such dataset does not exist {dataset_name}")
    return 

def convert_to(lst, data_type):
    output = []

    for data in lst: 
        # data is in the form of [("equation",...), ("answer",...), ("questions",...)]
        d_dict = dict(data)
        new_data = []
        flag = True

        for key, value in d_dict.items():
            if key =="equation":
                converter = EquationConverter()
                converter.eqset(value)

                if data_type == "infix":
                    new_value = converter.expr_as_infix()
                elif data_type == "prefix":
                    new_value = converter.expr_as_prefix()
                elif data_type == "postfix":
                    new_value = converter.expr_as_postfix()

                if re.match(r"[a-z] = .*\d+.*", new_value):
                    new_data.append((key, new_value))
                else:
                    flag = False
            else:
                new_data.append((key, value))

        if flag:
            output.append(new_data)

    return output

def get_data(dataset_name, lst, data_type):
    pre = convert_to(lst, "prefix")
    pos = convert_to(lst, "postfix")
    inf = []

    # NOTE:
    save_data_to_binary(os.path.join(DIR_PATH,f"{data_type}/{data_type}_{dataset_name}_prefix.pkl"), pre)
    save_data_to_binary(os.path.join(DIR_PATH,f"{data_type}/{data_type}_{dataset_name}_postfix.pkl"), pos)
    save_data_to_binary(os.path.join(DIR_PATH,f"{data_type}/{data_type}_{dataset_name}_infix.pkl"), inf)

    return pre, pos, inf
    
if __name__=="__main__":

    if not os.path.exists("./train/"):
        os.makedirs("./train/")

    if not os.path.exists("./test/"):
        os.makedirs("./test/")

    transform_dataset("MaWPS")
    transform_dataset("CommonCore")
    transform_dataset("Illinois")

    commoncore_test = commoncore[:int(len(commoncore)*test_split)]
    commoncore = commoncore[int(len(commoncore)*test_split):]

    illinois_test = illinois[:int(len(illinois)*test_split)]
    illinois = illinois[int(len(illinois)*test_split):]

    mawps_test = mawps[:int(len(mawps)*test_split)]
    mawps = mawps[int(len(mawps)*test_split):]

    random.shuffle(commoncore)
    random.shuffle(illinois)
    random.shuffle(mawps)

    problem_list = commoncore + illinois + mawps

    random.shuffle(problem_list)    
    # TODO: Incomplete 

    # CREATING TEST SET FOR ALL THE DATSETS

    pre_mawps_test, pos_mawps_test, inf_mawps_test = get_data("MaWPS", mawps_test, "test")
    pre_illinois_test, pos_illinois_test, inf_illinois_test = get_data("Illinois", illinois_test, "test")
    pre_commoncore_test, pos_commoncore_test, inf_commoncore_test = get_data("CommonCore", commoncore_test, "test")

    # CREATING TRAIN SET FOR ALL THE DATSETS
    pre_mawps_train, pos_mawps_train, inf_mawps_train = get_data("MaWPS", mawps, "train")
    pre_illinois_train, pos_illinois_train, inf_illinois_train = get_data("Illinois", illinois, "train")
    pre_commoncore_train, pos_commoncore_train, inf_commoncore_train = get_data("CommonCore", commoncore, "train")

    pre_all_test = pre_mawps_test+ pre_illinois_test + pre_commoncore_test
    pos_all_test = pos_mawps_test+ pos_illinois_test + pos_commoncore_test
    inf_all_test = inf_mawps_test+ inf_illinois_test + inf_commoncore_test


    pre_all_train = pre_mawps_train+ pre_illinois_train + pre_commoncore_train
    pos_all_train = pos_mawps_train+ pos_illinois_train + pos_commoncore_train
    inf_all_train = inf_mawps_train+ inf_illinois_train + inf_commoncore_train

    save_data_to_binary(os.path.join(DIR_PATH,f"test/test_all_prefix.pkl"), pre_all_test)
    save_data_to_binary(os.path.join(DIR_PATH,f"test/test_all_postfix.pkl"), pos_all_test)
    save_data_to_binary(os.path.join(DIR_PATH,f"test/test_all_infix.pkl"), inf_all_test)

    save_data_to_binary(os.path.join(DIR_PATH,f"train/train_all_prefix.pkl"), pre_all_train)
    save_data_to_binary(os.path.join(DIR_PATH,f"train/train_all_postfix.pkl"), pos_all_train)
    save_data_to_binary(os.path.join(DIR_PATH,f"train/train_all_infix.pkl"), inf_all_train)


    print("done")