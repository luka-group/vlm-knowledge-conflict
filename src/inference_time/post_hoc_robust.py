import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter

from src.utils.data_utils import load_dataset

def read_file(file_path):
    preds = {}
    with open(file_path, "r") as fin:
        for line in fin.readlines():
            preds.update(json.loads(line))
    return preds

def find_final_answer(matrix):
    # Convert the matrix to a numpy array for easier manipulation
    matrix = np.array(matrix)
    
    # Step 1: Extract the highest scores for each sample
    highest_scores_indices = np.argmax(matrix, axis=1)
    
    # Step 2: Count the frequency of each label
    label_counts = Counter(highest_scores_indices)
    
    # Step 3: Determine the most frequent label
    final_answer = label_counts.most_common(1)[0][0]
    
    return final_answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    # parser.add_argument("--aggregate_method", choices=["vote", "var"])
    parser.add_argument("--score_file_textual", type=str)
    parser.add_argument("--score_file_visual", type=str)
    args = parser.parse_args()
    
    textual_preds = read_file(args.score_file_textual)
    visual_preds = read_file(args.score_file_visual)
    textual_name = args.score_file_textual.split("/")[-1]
    visual_name = args.score_file_visual.split("/")[-1]
    
    # load dataset
    dataset_nickname, dataset = load_dataset(args)
            
    pb = tqdm(range(len(dataset)))
    count_textual = 0
    count_visual = 0
    
    for data in dataset:
        data_id = data["id"]
        textual_pred = textual_preds.get(data_id)
        visual_pred = visual_preds.get(data_id)
        textual_scores = [pred[1] for pred in textual_pred]
        visual_scores = [pred[1] for pred in visual_pred]
        textual_answer = find_final_answer(textual_scores)
        visual_answer = find_final_answer(visual_scores)
        textual_selected_scores = [score[textual_answer] for score in textual_scores]
        visual_selected_scores = [score[visual_answer] for score in visual_scores]
        
        # textual_var = np.var(textual_selected_scores)
        # visual_var = np.var(visual_selected_scores)
        
        textual_var = np.var(textual_scores)
        visual_var = np.var(visual_scores)
        
        if textual_var < visual_var:
            final_answer = chr(ord("A") + textual_answer)
            count_textual += 1
        else:
            final_answer = chr(ord("A") + visual_answer)
            count_visual += 1
        
        with open(f"outputs/inference_time/robust_{args.dataset}_{textual_name}.{visual_name}.txt", "a+") as fout:
            fout.write(f"{json.dumps({data_id: final_answer})}\n")
        pb.update(1)
    
    print(f"Textual selected: {count_textual}")
    print(f"Visual selected: {count_visual}")
        
if __name__ == "__main__":
    main()