import os
import base64
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image_path = inputs["image_path"]
  
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    image_base64 = encode_image(image_path)
    return {"image": image_base64}

class APIChat():
    def __init__(self, config, prompt="") -> None:
        # setup configure
        self.is_local = False
        self.config = config
        self.prompt = prompt
        
        # load model
        if "gpt" in config.model_name.lower():
            self.model = ChatOpenAI(
                model = config.model_name,
                temperature = config.temperature,
                max_tokens = config.max_length,
                top_k=config.top_k,
                top_p=config.top_p,
                n=config.sampling_times)
        elif "gemini" in config.model_name.lower():
            self.model = ChatGoogleGenerativeAI(
                model = config.model_name,
                temperature = config.temperature,
                max_output_tokens = config.max_length,
                top_k=config.top_k,
                top_p=config.top_p,
                n=config.sampling_times
            )
    
    def chat(self, **kwargs):
        if self.mode == "text":
            text = kwargs.text
            messages = []
            if len(self.prompt) > 0:
                for t in text:
                    messages.append([
                        SystemMessage(content=self.prompt),
                        HumanMessage(content=t),
                    ])
            else:
                for t in text:
                    messages.append([
                        HumanMessage(content=t),
                    ])
            outputs = self.model.invoke(messages)
                
        elif self.mode == "visual":
            text = inputs.text
            images = inputs.image
            inputs = self.processor(images=images, text=text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                **self.generate_config, 
            )
                
        return outputs