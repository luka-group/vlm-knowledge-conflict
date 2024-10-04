from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser()
    
    # model & inference
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sampling_times", type=int, default=1)
    
    # data
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--is_fact_given", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    return parser