import os
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
# from vllm.config import VisionLanguageConfig

from src.utils.data_utils import load_dataset
from src.utils.parser_utils import get_parser
from src.prompt import qa_prompt, qa_context_prompt, qa_image_prompt, qa_blend_prompt

llava_visual_format = """"{system_prompt}\nUSER: <image>\n{user_input}\nASSISTANT:"""
llava_textual_format = """"{system_prompt}\nUSER: {user_input}\nASSISTANT:"""
llava_34b_textual_format = """"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"""
llava_34b_visual_format = """"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<image>\n{user_input}<|im_end|>\n<|im_start|>assistant\n"""
blip_format = """"{system_prompt}\n{user_input}Answer:"""

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
    parser.add_argument("--method", choices=["post_hoc", "prob"])
    parser.add_argument("--conflict_prompt", choices=["fixed", "answer", "prob"])
    parser.add_argument("--prob_method", choices=["max", "compare"])
    args = parser.parse_args()
    
    if args.greedy:
        args.temperature = 0.0
        
    model_nickname = args.model_name.split("/")[-1]
    
    # load dataset
    dataset_nickname, dataset = load_dataset(args)
    model_nickname = args.model_name.split("/")[-1]
    text_preds = {}
    with open(f"outputs/analysis/{dataset_nickname}/{model_nickname}/{args.dataset}_textual_T0.0.txt.score", "r") as fin:
        for line in fin.readlines():
            text_preds.update(json.loads(line))
    visual_preds = {}
    with open(f"outputs/analysis/{dataset_nickname}/{model_nickname}/{args.dataset}_visual_T0.0.txt.score", "r") as fin: # elicit_{args.dataset}_post_hoc.txt    {args.dataset}_textual_T0.0.txt.score
        for line in fin.readlines():
            visual_preds.update(json.loads(line))

    output_dir = f"outputs/inference_time/{dataset_nickname}/{model_nickname}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, f"{args.dataset}_{args.method}_{args.conflict_prompt}_{args.prob_method}_T{args.temperature}.txt")
    
    pb = tqdm(range(len(dataset)))
    
    if args.method == "post_hoc":
        model = LLM(
            model=args.model_name,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        
        sampling_params = SamplingParams(
            n=1,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        
        for data in dataset:
            data_id = data["id"]
            question = data["question"]
            choices = data["multiple_choices"]
            choices_text = ""
            for c_name, c_content in choices.items():
                choices_text += f"{c_name}: {c_content}\n"
            text = f"Question:\n{question}\nOption:\n{choices_text}"

            image_path = os.path.join(f"data/{dataset_nickname}/images", data["image"])
            if not os.path.exists(image_path):
                pb.update(1)
                continue
                
            text_pred = text_preds.get(data_id)
            text_answer = clean_answer(choices, text_pred)
            visual_pred = visual_preds.get(data_id)
            visual_answer = clean_answer(choices, visual_pred)

            if text_answer != visual_answer:
                if args.conflict_prompt == "fixed":
                    conflict_prompt = "Be aware that your visual memory might differ from your text memory, causing a conflict in your knowledge.\n"
                    text = text + conflict_prompt
                elif args.conflict_prompt == "answer":
                    conflict_prompt = f"Be aware that your visual memory might differ from your text memory, causing a conflict in your knowledge. Your text memory is: {text_answer} and your visual memory is: {visual_answer}.\n"
                    text = text + conflict_prompt
            
            if "34b" in model_nickname:
                text = llava_34b_visual_format.format(system_prompt=qa_prompt, user_input=text)
            elif "blip" in model_nickname:
                text = blip_format.format(system_prompt=qa_prompt, user_input=text)
            else:
                text = llava_visual_format.format(system_prompt=qa_prompt, user_input=text)
            
            
            image = Image.open(image_path)
            answer = model.generate(
                {
                    "prompt": text,
                    "multi_modal_data": {
                        "image": image
                    }
                },
                sampling_params=sampling_params
            )
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: answer[0].outputs[0].text})}\n")
            pb.update(1)

if __name__ == "__main__":
    main()            