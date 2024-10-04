# from src.models.api import APIChat
from src.models.local import LocalModelChat

def get_model(args):
    api_models = ["gpt", "gemini", "claude"]
    local_models = ["blip", "llava", "vicuna", "qwen"]
    # for model in api_models:
    #     if model in args.model_name:
    #         return APIChat
    for model in local_models:
        if model in args.model_name.lower():
            return LocalModelChat