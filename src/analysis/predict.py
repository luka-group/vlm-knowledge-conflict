import os
import json
from PIL import Image
from tqdm import tqdm

from src.utils.data_utils import load_dataset
from src.models import get_model
from src.utils.parser_utils import get_parser
from src.prompt import qa_prompt, qa_context_prompt, qa_image_prompt, qa_blend_prompt

def main():
    parser = get_parser()
    parser.add_argument("--is_scored", action="store_true")
    args = parser.parse_args()
    
    if args.greedy:
        args.temperature = 0.0
        
    # load dataset
    dataset_nickname, dataset = load_dataset(args)
    model_nickname = args.model_name.split("/")[-1]
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
    
    model = get_model(args)(args, prompt=qa_prompt)
    if mode == "textual":
        model.remode("text")

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
        if mode == "visual":
            image = Image.open(os.path.join(f"data/{dataset_nickname}/images", data["image"]))
            context = {"text": text, "image": image}
        elif mode == "recognize":
            image = Image.open(os.path.join(f"data/{dataset_nickname}/images", data["image"]))
            context = {"text": "What/Who is in the image? Do not describe details. Just give a named entity, e.g. Jackie Chan.", "image": image}
        context.update({"is_scored": args.is_scored})
        answer = model.chat(**context)
        with open(output_path, "a+") as fout:
            fout.write(f"{json.dumps({data_id: answer})}\n")
        pb.update(1)

if __name__ == "__main__":
    main()            