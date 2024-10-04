import os
import json
from PIL import Image
from tqdm import tqdm

from src.utils.data_utils import load_dataset
from src.utils.parser_utils import get_parser
from src.models.contrastive_decoding import ContrastiveDecodingModel
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

def main():
    parser = get_parser()
    parser.add_argument("--visual_type", type=str, choices=["visual", "visual_ori", "visual_ori_blank"])
    parser.add_argument("--kl", type=float, default=-1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--is_scored", action="store_true")
    parser.add_argument("--robust", default=None)
    args = parser.parse_args()
    if args.greedy:
        args.temperature = 0.0
    
    # load dataset
    dataset_nickname, dataset = load_dataset(args)
    model = ContrastiveDecodingModel(args, qa_prompt)
    if args.robust is None:
        output_path = os.path.join(args.output_dir, f"elicit_{args.dataset}_{args.model_name}_{args.visual_type}_KL{args.kl}_alpha{args.alpha}_beta{args.beta}.txt")
    else:
        output_path = os.path.join(args.output_dir, f"elicit_sampled_{args.dataset}_{args.model_name}_{args.visual_type}_KL{args.kl}_alpha{args.alpha}_beta{args.beta}.txt")
        
    if args.is_scored:
        output_path += ".score"

    pb = tqdm(range(len(dataset)))
    
    if args.robust is None:
        for data in dataset:
            data_id = data["id"]
            question = data["question"]
            choices = data["multiple_choices"]
            choices_text = ""
            for c_name, c_content in choices.items():
                choices_text += f"{c_name}: {c_content}\n"
            if "blank" in args.visual_type:
                image = Image.new('RGB', (336, 336), color = (255,255,255))
            else:
                image = Image.open(os.path.join("data/viquae/images", data["image"]))

            entity = data.get(entity)
            if entity is None:
                caption = ""
            else:
                caption = f"This is an image of {entity}."
            text_for_text_logit = f"{caption}\nQuestion: {question}\nOption:\n{choices_text}"
            text_for_visual_logit = f"Question: {question}\nOption:\n{choices_text}"
            answer = model.generate(
                text_for_text_logit=text_for_text_logit,
                text_for_visual_logit=text_for_visual_logit,
                image=image,
                is_scored=args.is_scored
            )
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: answer})}\n")
            pb.update(1)
    else:
        args.is_scored = True
        for data in dataset:
            data_id = data["id"]
            question = data["input"]
            ori_question = data["original_question"]
            choices = data["multiple_choices"]
            choices_text = ""
            for c_name, c_content in choices.items():
                choices_text += f"{c_name}: {c_content}\n"
            if "blank" in args.visual_type:
                image = Image.new('RGB', (336, 336), color = (255,255,255))
            else:
                image = Image.open(os.path.join("data/viquae/images", data["image"]))
            
            entity = data.get(entity)
            if entity is None:
                caption = ""
            else:
                caption = f"This is an image of {entity}."
            text_for_text_logit = f"{caption}\nQuestion: {question}\nOption:\n{choices_text}"
            text_for_visual_logit = f"Question: {question}\nOption:\n{choices_text}"
            
            answers = []
            for prompt in PROMPTS:
                model.prompt = prompt + "\n"
                answer = model.generate(
                    text_for_text_logit=text_for_text_logit,
                    text_for_visual_logit=text_for_visual_logit,
                    image=image,
                    is_scored=args.is_scored
                )
                answers.append(answer)
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: answers})}\n")
            pb.update(1)
        
if __name__ == "__main__":
    main()            