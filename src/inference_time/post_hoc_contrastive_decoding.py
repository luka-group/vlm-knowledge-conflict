import re
import os
import json
import torch
import numpy as np
from tqdm import tqdm

from src.utils.data_utils import load_dataset
from src.utils.parser_utils import get_parser

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
    
def get_answer_and_prob(options, pred):
    answer = clean_answer(options, pred[0])
    if answer is None:
        prob = max(pred[1])
        answer = chr(ord("A") + np.argmax(pred[1]))
    else:
        prob = pred[1][ord(answer) - ord("A")]
    return answer, prob

def main():
    parser = get_parser()
    parser.add_argument("--method", choices=["add", "dynamic"])
    args = parser.parse_args()
    if args.greedy:
        args.temperature = 0.0
        
    model_nickname = args.model_name.split("/")[-1]
    
    # load dataset
    dataset_nickname, dataset = load_dataset(args)

    
    text_preds = {}
    with open(f"outputs/analysis/{dataset_nickname}/{model_nickname}/{args.dataset}_textual_T0.0.txt.score", "r") as fin:
        for line in fin.readlines():
            text_preds.update(json.loads(line))
    
    # for dynamic cd, we need to load the original visual answer score
    if args.method == "add":        
        visual_preds = {}
        with open(f"outputs/analysis/{dataset_nickname}/{model_nickname}/elicit_{args.dataset}_post_hoc.txt", "r") as fin: # elicit_{args.dataset}_post_hoc.txt    {args.dataset}_textual_T0.0.txt.score
            for line in fin.readlines():
                visual_preds.update(json.loads(line))
    elif args.method == "dynamic":
        visual_preds = {}
        with open(f"outputs/analysis/{dataset_nickname}/{model_nickname}/{args.dataset}_visual_T0.0.txt.score", "r") as fin: # elicit_{args.dataset}_post_hoc.txt    {args.dataset}_textual_T0.0.txt.score
            for line in fin.readlines():
                visual_preds.update(json.loads(line))
                
    output_dir = f"outputs/inference_time/{dataset_nickname}/{model_nickname}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    pb = tqdm(range(len(dataset)))
    for data in dataset:
        data_id = data["id"]
        # print(data["multiple_choices_answer"])
        choices = data["multiple_choices"]
        text_pred = text_preds.get(data_id)
        visual_pred = visual_preds.get(data_id)
        
        if text_pred is None or visual_pred is None:
            print(data_id)
            continue
        
        text_answer = clean_answer(choices, text_pred)
        visual_answer = clean_answer(choices, visual_pred)
        
        if text_answer == visual_answer:
            with open(f"outputs/inference_time/{dataset_nickname}/{model_nickname}/{args.dataset}_prob_{args.method}.txt", "a+") as fout:
                fout.write(f"{json.dumps({data_id: text_answer})}\n")
            pb.update(1)
            continue
        
        # print(f"Textual answer: {text_pred[0].strip()}")
        # print(f"Visual answer: {visual_pred[0].strip()}")
        
        text_logit = torch.tensor(text_pred[1])
        visual_logit = torch.tensor(visual_pred[1])
        
        # print(text_logit)
        # print(visual_logit)
        
        text_prob = torch.nn.functional.softmax(text_logit)
        
        # for original visual score, we need to first calculate its probability
        if args.method == "dynamic":
            visual_prob = torch.nn.functional.softmax(visual_logit)
        else:
            visual_prob = visual_logit
        
        max_text_prob = torch.max(text_prob)
        max_visual_prob = torch.max(visual_prob)
        
        if args.method == "add":
            cd_logit = max_visual_prob * visual_prob + max_text_prob * text_prob
        elif args.method == "dynamic":
            if max_text_prob > max_visual_prob:
                cd_logit = max_text_prob * text_logit - max_visual_prob * visual_logit
            else:
                cd_logit = max_visual_prob * visual_logit - max_text_prob * text_logit
        
        # print(cd_logit)
        
        cd_prob = torch.nn.functional.softmax(cd_logit)
        cd_prob_index = torch.argmax(cd_prob)
        answer = chr(ord("A") + cd_prob_index)
        
        # print(answer)
        # input()
        
        with open(f"outputs/inference_time/{dataset_nickname}/{model_nickname}/{args.dataset}_prob_{args.method}.txt", "a+") as fout:
            fout.write(f"{json.dumps({data_id: answer})}\n")
        
        pb.update(1)
        
if __name__ == "__main__":
    main()