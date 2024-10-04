import re
import json
import math
import torch
import numpy as np
from tqdm import tqdm

from src.utils.parser_utils import get_parser
from src.utils.data_utils import load_dataset

def clean_answer(options, answer):
    if type(answer) is list:
        answer = answer[0]
    pattern = r"\b[A-D]\b|[A-D](?=\s|:)"
    match = re.search(pattern, answer) 
    if match is None:   
        for option, content in options.items():
            if content in answer:
                return option
        return None
    else:
        return match.group()

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.greedy:
        args.temperature = 0.0
        
    model_nickname = args.model_name.split("/")[-1]
    
    # load dataset
    dataset_nickname, dataset = load_dataset(args)
    text_preds = {}
    with open(f"outputs/analysis/{dataset_nickname}/{model_nickname}/{dataset_nickname}_mc_textual_T0.0.txt.score", "r") as fin:
        for line in fin.readlines():
            text_preds.update(json.loads(line))
    visual_preds = {}
    with open(f"outputs/analysis/{dataset_nickname}/{model_nickname}/{dataset_nickname}_mc_visual_T0.0.txt.score", "r") as fin:
        for line in fin.readlines():
            visual_preds.update(json.loads(line))
            
    
    pb = tqdm(range(len(dataset)))
    sum_cd_wo_conflict = 0
    sum_cd_conflict = 0
    sum_cd = 0
    
    cd_wo_conflicts = []
    cd_conflicts = []
    cds = []
    
    cnt_wo_conflict = 0
    cnt_conflict = 0
    cnt_valid_sample = 0
    
    for data in dataset:
        data_id = data["id"]
        choices = data["multiple_choices"]
        answer = data["multiple_choices_answer"]
        answer_index = ord(answer) - ord("A")
        text_pred = text_preds.get(data_id)
        visual_pred = visual_preds.get(data_id)
        if text_pred is None or visual_pred is None:
            # print(data_id)
            continue
        
        text_answer = clean_answer(choices, text_pred)
        visual_answer = clean_answer(choices, visual_pred)            
        
        text_prob = torch.nn.functional.softmax(torch.tensor(text_pred[1]))
        visual_prob = torch.nn.functional.softmax(torch.tensor(visual_pred[1]))
        cd_metric = torch.abs(torch.log(visual_prob / text_prob)[answer_index]).item()
        
        # print(cd_metric)
        # input()
        if cd_metric == np.inf or cd_metric == -np.inf or math.isnan(cd_metric):
            # print(data_id)
            continue
        
        if text_answer == visual_answer:
            sum_cd_wo_conflict += cd_metric
            cnt_wo_conflict += 1
            cd_wo_conflicts.append(cd_metric)
        else:
            sum_cd_conflict += cd_metric
            cnt_conflict += 1
            cd_conflicts.append(cd_metric)
        
        sum_cd += cd_metric
        cnt_valid_sample += 1
        cds.append(cd_metric)
        
        pb.update(1)
        
    print(f"CD Metric Avg.: {sum_cd / cnt_valid_sample}")
    print(f"CD Metric w/o conflict: {sum_cd_wo_conflict / cnt_wo_conflict}")
    print(f"CD Metric conflict: {sum_cd_conflict / cnt_conflict}")
    
    with open(f"outputs/draw/{dataset_nickname}_{model_nickname}_scores.txt", "w") as fout:
        fout.write(f"{json.dumps(cds)}\n")
        fout.write(f"{json.dumps(cd_wo_conflicts)}\n")
        fout.write(f"{json.dumps(cd_conflicts)}\n")
        
if __name__ == "__main__":
    main()