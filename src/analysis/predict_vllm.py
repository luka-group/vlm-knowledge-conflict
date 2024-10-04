import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.utils.data_utils import load_dataset
from src.utils.parser_utils import get_parser
from src.prompt import qa_prompt, qa_context_prompt, qa_image_prompt, qa_blend_prompt

llava_visual_format = """"{system_prompt}\nUSER: <image>\n{user_input}\nASSISTANT:\n"""
llava_textual_format = """"{system_prompt}\nUSER: {user_input}\nASSISTANT:\n"""
llava_34b_textual_format = """"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"""
llava_34b_visual_format = """"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<image>\n{user_input}<|im_end|>\n<|im_start|>assistant\n"""
blip_format = """"{system_prompt}\n{user_input}Answer:"""

def main():
    parser = get_parser()
    parser.add_argument("--is_scored", action="store_true")
    args = parser.parse_args()
    
    if args.greedy:
        args.temperature = 0.0
        
    model_nickname = args.model_name.split("/")[-1]
    
    # load dataset
    dataset_nickname, dataset = load_dataset(args)

    if "textual" in args.dataset:
        mode = "textual"
    elif "visual" in args.dataset:
        mode = "visual"
    elif "recognize" in args.dataset:
        mode = "recognize"
        
    output_dir = os.path.join(args.output_dir, model_nickname)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = f"outputs/analysis/{dataset_nickname}/{model_nickname}/{args.dataset}_{mode}.txt"
    
    if args.is_scored:
        output_path += ".score"
    
    model = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        logprobs=20,
    )

    pb = tqdm(range(len(dataset)))
    for data in dataset:
        data_id = data["id"]
        question = data["question"]
        choices = data["multiple_choices"]
        choices_text = ""
        for c_name, c_content in choices.items():
            choices_text += f"{c_name}: {c_content}\n"
        text = f"Question:\n{question}\nOption:\n{choices_text}"
        
        if mode == "textual":
            entity = data.get(entity)
            if entity is None:
                caption = ""
            else:
                caption = f"This is an image of {entity}."
            text = caption + "\n" + text
            if "34b" in model_nickname:
                text = llava_34b_textual_format.format(system_prompt=qa_prompt, user_input=text)
            elif "blip" in model_nickname:
                text = blip_format.format(system_prompt=qa_prompt, user_input=text)
            else:
                text = llava_textual_format.format(system_prompt=qa_prompt, user_input=text)
        elif mode == "visual":
            if "34b" in model_nickname:
                text = llava_34b_visual_format.format(system_prompt=qa_prompt, user_input=text)
            elif "blip" in model_nickname:
                text = blip_format.format(system_prompt=qa_prompt, user_input=text)
            else:
                text = llava_visual_format.format(system_prompt=qa_prompt, user_input=text)
        elif mode == "recognize":
            text = "What/Who is in the image? Do not describe details. Just give a named entity, e.g. Jackie Chan."
            if "34b" in model_nickname:
                text = llava_34b_visual_format.format(system_prompt="", user_input=text)
            elif "blip" in model_nickname:
                text = blip_format.format(system_prompt="", user_input=text)
            else:
                text = llava_visual_format.format(system_prompt="", user_input=text)
            
        if mode == "visual" or mode == "recognize":
            image = Image.open(os.path.join(f"data/{dataset_nickname}/images", data["image"]))
            answer = model.generate(
                {
                    "prompt": text,
                    "multi_modal_data": {
                        "image": image
                    }
                },
                sampling_params=sampling_params
            )
        else:
            answer = model.generate(
                text,
                sampling_params=sampling_params
            )
        if args.is_scored:
            target_tokens = ["A", "B", "C", "D"]
            target_scores = [-np.inf, -np.inf, -np.inf, -np.inf]
            logprobs = answer[0].outputs[0].logprobs[0]
            for _, logprob in logprobs.items():
                decoded_token = logprob.decoded_token.strip()
                try:
                    target_index = target_tokens.index(decoded_token)
                    target_scores[target_index] = logprob.logprob
                except:
                    continue
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: [answer[0].outputs[0].text, target_scores]})}\n")
        else:
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: answer[0].outputs[0].text})}\n")
        pb.update(1)

if __name__ == "__main__":
    main()            