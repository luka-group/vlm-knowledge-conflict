import re
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from argparse import ArgumentParser

def read_pred_file(input_file):
    preds = {}
    with open(input_file, "r") as fin:
        for line in fin.readlines():
            preds.update(json.loads(line))
    return preds

def clean_answer(options, answer):
    for option, content in options.items():
        # if option not in answer:
        #     if answer.lower() in content.lower():
        #         return option
        # else: 
        answer = answer.replace("Option", "")
        answer = answer.split(":")[0]
        if option in answer:
            return option
    return None

def match(options, gold_answer, answer):
    if type(answer) is list:
        # try:
        #     potential_answer = chr(ord("A") + np.argmax(answer[1]))
        # except:
        #     potential_answer = None
        answer = answer[0]
    # # cleaned_answer = clean_answer(options, answer)
    # # if cleaned_answer is None:
    # #     return False
    # # if cleaned_answer == gold_answer:
    # #     return True
    # # return False
    # if gold_answer not in answer:
    #     gold_content = options[gold_answer]
    #     if answer.lower() == gold_content.lower():
    #         return True
    #     # print(answer)
    #     # input()
    #     return False
    # answer = answer.replace("Option", "")
    # if ":" not in answer:
    #     return True
    # answer = answer.split(":")[0]
    # if gold_answer in answer:
    #     return True
    # # print(answer)
    # # input()

    pattern = r"\b[A-D]\b|[A-D](?=\s|:)"
    match = re.search(pattern, answer)
    if match is None:
        # print(answer)
        # print(gold_answer)
        # input()
        # if potential_answer is None:
        if options[gold_answer] in answer:
            return True
        else:
            return False
        # else:
        #     if potential_answer == gold_answer:
        #         return True
        #     return False
    match = match.group()
    if match == gold_answer:
        return True
    return False

def compare(options, answer_1, answer_2):
    # ori_1 = answer_1
    # ori_2 = answer_2
    if type(answer_1) is list:
        # try:
        #     potential_answer_1 = chr(ord("A") + np.argmax(answer_1[1]))
        # except:
        #     potential_answer_1 = None
        answer_1 = answer_1[0]
    if type(answer_2) is list:
        # try:
        #     potential_answer_2 = chr(ord("A") + np.argmax(answer_2[1]))
        # except:
        #     potential_answer_2 = None
        answer_2 = answer_2[0]
    # print(answer_1)
    # print(answer_2)
    answer_1 = clean_answer(options, answer_1)
    answer_2 = clean_answer(options, answer_2)
    # print(answer_1)
    # print(answer_2)
    # input()
    # if answer_1 is None:
    #     answer_1 = potential_answer_1
    # if answer_2 is None:
    #     answer_2 = potential_answer_2
    if answer_1 is None or answer_2 is None:
        return False
    if answer_1 == answer_2:
        return True
    # print(ori_1)
    # print(ori_2)
    # input()
    return False
        

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--input_file_2", type=str, default=None)
    parser.add_argument("--cleaned_model", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.dataset == "viquae":
        if args.cleaned_model is not None:
            with open(f"data/viquae/cleaned_dataset_mc_{args.cleaned_model}.json", "r") as fin:
                dataset = json.load(fin)
        else:
            with open("data/viquae/multiple_choice_data.json", "r") as fin:
                dataset = json.load(fin)
    elif args.dataset == "infoseek":
        if args.cleaned_model is not None:
            with open(f"data/infoseek/llava_recognized_infoseek_val_mc.json", "r") as fin:
                dataset = json.load(fin)
        else:
            with open("data/infoseek/infoseek_val_mc.json", "r") as fin:
                dataset = json.load(fin)
        
    if args.input_file_2 is None:
        preds = read_pred_file(args.input_file)
        cnt_correct = 0
        cnt = 0
        for data in dataset:
            if args.dataset == "viquae":
                data_id = data["id"]
            elif args.dataset == "infoseek":
                data_id = data["data_id"]
            pred = preds.get(data_id)
            if pred is None:
                continue
            options = data["multiple_choices"]
            gold_answer = data["multiple_choices_answer"]
            flag = match(options, gold_answer, pred)
            if flag:
                cnt_correct += 1
            cnt += 1
        print(f"Accuracy: {cnt_correct / cnt}")
    else:
        preds_1 = read_pred_file(args.input_file)
        preds_2 = read_pred_file(args.input_file_2)
        cnt_conflict = 0
        cnt = 0
        for data in dataset:
            if args.dataset == "viquae":
                data_id = data["id"]
            elif args.dataset == "infoseek":
                data_id = data["data_id"]
            pred_1 = preds_1.get(data_id)
            pred_2 = preds_2.get(data_id)
            if pred_1 is None or pred_2 is None:
                continue
            options = data["multiple_choices"]
            flag = compare(options, pred_1, pred_2)
            if not flag:
                cnt_conflict += 1
            cnt += 1
        print(f"Conflict Rate: {cnt_conflict / cnt}")

if __name__ == "__main__":
    main()   