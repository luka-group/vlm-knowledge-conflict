import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.data_utils import load_dataset
from src.utils.parser_utils import get_parser

def main():
    parser = get_parser()
    parser.add_argument("--method", choices=["compare", "shift"])
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

    output_dir = os.path.join(args.output_dir, dataset_nickname)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, model_nickname)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, f"{args.dataset}_prob_{args.method}.txt")

    pb = tqdm(range(len(dataset)))
    for data in dataset:
        data_id = data["id"]
        text_pred = text_preds.get(data_id)
        visual_pred = visual_preds.get(data_id)
        
        if text_pred is None or visual_pred is None:
            continue
        
        text_prob = text_pred[1]
        visual_prob = visual_pred[1]
        text_prob = torch.nn.functional.softmax(torch.tensor(text_prob)).tolist()
        visual_prob = torch.nn.functional.softmax(torch.tensor(visual_prob)).tolist()
        
        if args.method == "compare":
            if max(text_prob) > max(visual_prob):
                answer = chr(ord("A") + np.argmax(text_prob))
            else:
                answer = chr(ord("A") + np.argmax(visual_prob))
            # print(answer)
            # input()
        elif args.method == "shift":
            shift_prob = np.array(visual_prob) - np.array(text_prob)
            answer = chr(ord("A") + np.argmax(shift_prob))
        with open(output_path, "a+") as fout:
            fout.write(f"{json.dumps({data_id: answer})}\n")
        pb.update(1)

if __name__ == "__main__":
    main() 