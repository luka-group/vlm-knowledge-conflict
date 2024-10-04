import json
import torch
import numpy as np
from tqdm import tqdm

from src.utils.data_utils import load_dataset
from src.utils.parser_utils import get_parser

def main():
    parser = get_parser()
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
    visual_preds = {}
    with open(f"outputs/analysis/{dataset_nickname}/{model_nickname}/{args.dataset}_visual_T0.0.txt.score", "r") as fin:
        for line in fin.readlines():
            visual_preds.update(json.loads(line))

    pb = tqdm(range(len(dataset)))
    for data in dataset:
        data_id = data["id"]
        text_pred = text_preds.get(data_id)
        visual_pred = visual_preds.get(data_id)
        
        text_prob = torch.tensor(text_pred[1])
        visual_prob = torch.tensor(visual_pred[1])
        
        cd_prob = visual_prob - text_prob
        cd_prob = torch.nn.functional.softmax(cd_prob)
        cd_prob_index = torch.argmax(cd_prob)
        answer = chr(ord("A") + cd_prob_index)
        
        with open(f"outputs/analysis/{dataset_nickname}/{model_nickname}/elicit_{args.dataset}_post_hoc.txt", "a+") as fout:
            fout.write(f"{json.dumps({data_id: [answer, cd_prob.tolist()]})}\n")
        
        pb.update(1)
        
if __name__ == "__main__":
    main()