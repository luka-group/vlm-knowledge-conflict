import os
import json
import torch
from PIL import Image
from tqdm import tqdm

from src.models import get_model
from src.utils.data_utils import load_dataset
from src.utils.parser_utils import get_parser
from src.prompt import qa_prompt, qa_context_prompt, qa_image_prompt, qa_blend_prompt

PROMPTS = [
    "As a specialist in answering queries, please provide the direct answer to the question without any additional explanation or follow-up questions.",
    "Please use your expertise in question answering to give the answer directly. Avoid offering any explanations or further inquiries.",
    "Relying on your knowledge of answering questions, output the answer succinctly without including any extra information or asking further questions.",
    "With your proficiency in question answering, provide just the answer to the given question. Do not include any explanations or follow-up questions.",
    "As a question-answering expert, your task is to give the answer only. Refrain from offering any explanations or asking subsequent questions.",
    "Given your expertise in providing answers, please offer the answer directly to the question, and do not include further explanations or additional questions.",
    "Use your proficiency in answering questions to supply only the answer. Avoid explaining or posing any further questions.",
    "Utilizing your skill in question answering, please deliver the answer right away. No explanations or further questions should be included.",
    "With your experience in giving direct answers, provide the answer to the question without elaboration or posing additional questions.",
    "As an adept in question answering, please produce only the answer to the question and omit any explanations or further",
]

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

def monte_carlo_predictions(model, inputs, n_samples):
    model.train()  # Set the model to training mode to enable dropout
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            outputs = model(**inputs)
            predictions.append(outputs.logits.unsqueeze(0))
    return torch.cat(predictions)

def main():
    parser = get_parser()
    parser.add_argument("--uncertainty_method", choices=["prompt", "dropout"])
    parser.add_argument("--is_scored", action="store_true")
    args = parser.parse_args()
    
    if args.greedy:
        args.temperature = 0.0
        
    # load dataset
    dataset_nickname, dataset = load_dataset(args)
    model_nickname = args.model_name.split("/")[-1]
    
    if "textual" in args.dataset:
        prompt = qa_prompt
        mode = "textual"
    elif "visual" in args.dataset:
        prompt = qa_prompt
        mode = "visual"
    
    output_dir = os.path.join(args.output_dir, model_nickname)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = f"outputs/analysis/{dataset_nickname}/{model_nickname}/{args.dataset}_{args.uncertainty_method}_T{args.temperature}.txt"
    
    if args.is_scored:
        output_path += ".score"
    
    model = get_model(args)(args, prompt=prompt)
    if "textual" in mode:
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
        
        if args.uncertainty_method == "prompt":
            answers = []
            for prompt in PROMPTS:
                model.prompt = prompt
                if mode == "visual":
                    image = Image.open(os.path.join("data/viquae/images", data["image"]))
                    context = {"text": text, "image": image}
                elif "textual" in mode:
                    context = {"text": text}
                context.update({"is_scored": args.is_scored})
                answer = model.chat(**context)
                answers.append(answer)
                
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: answers})}\n")
            pb.update(1)
            
        elif args.uncertainty_method == "dropout":
            # enable_dropout(model.model)
            model.model.train()
            model.model.training = True
            for layer in model.model.language_model.model.layers:
                layer.self_attn.attention_dropout = 0.1
            answers = []
            for _ in range(10):
                if mode == "visual":
                    image = Image.open(os.path.join("data/viquae/images", data["image"]))
                    context = {"text": text, "image": image}
                elif "textual" in mode:
                    context = {"text": text}
                context.update({"is_scored": args.is_scored})
                answer = model.chat(**context)
                answers.append(answer)

            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: answers})}\n")
            pb.update(1)

if __name__ == "__main__":
    main()            