import json
import torch
from argparse import ArgumentParser
from torch.nn.functional import kl_div

def read_pred_file(input_file):
    preds = {}
    with open(input_file, "r") as fin:
        for line in fin.readlines():
            preds.update(json.loads(line))
    return preds

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file_1", type=str)
    parser.add_argument("--input_file_2", type=str)
    
    args = parser.parse_args()
    
    preds_1 = read_pred_file(args.input_file_1)
    preds_2 = read_pred_file(args.input_file_2)
    
    avg_kl = 0
    cnt = 0
    for key in preds_1.keys():
        pred_1 = preds_1.get(key)
        pred_2 = preds_2.get(key)
        if pred_2 is None:
            continue
        # kl = (kl_div(torch.tensor(pred_1[1]), torch.tensor(pred_2[1])) + kl_div(torch.tensor(pred_2[1]), torch.tensor(pred_1[1]))) / 2
        kl = kl_div(torch.tensor(pred_2[1]), torch.tensor(pred_1[1]))
        file_1 = args.input_file_1.split("/")[-1]
        file_2 = args.input_file_2.split("/")[-1]
        with open(f"outputs/analysis/{file_1}_VS_{file_2}", "a+") as fout:
            fout.write(f"{json.dumps({key: kl.item()})}\n")
        avg_kl += kl.item()
        cnt += 1
    print(f"Avg. KL: {avg_kl / cnt}")

if __name__ == "__main__":
    main()