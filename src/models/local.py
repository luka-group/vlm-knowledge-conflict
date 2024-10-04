import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

llava_visual_format = """"{system_prompt}\nUSER: <image>\n{user_input}\nASSISTANT:"""
blip_visual_format = """"{system_prompt}\n{user_input}Answer:"""
text_format = """"{system_prompt}\nUSER: {user_input}\nASSISTANT:"""

class LocalModelChat():
    def __init__(self, config, prompt=""):
        # setup configure
        self.is_local = True
        self.config = config
        self.prompt = prompt
        self.device = torch.device(config.device)
        self.generate_config = {
            "do_sample": True,
            "max_new_tokens": config.max_new_tokens,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "length_penalty": config.length_penalty,
            "temperature": config.temperature,
        }
        if self.config.greedy:
            self.generate_config.update({"do_sample": False})
        
        # load model to device
        if "blip" in config.model_name.lower():
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.mode = "visual"
            self.prompt_format = blip_visual_format
            self.num_visual_tokens = int((self.model.config.vision_config.image_size / self.model.config.vision_config.patch_size) ** 2)
        elif "llava" in config.model_name.lower():
            self.processor = LlavaProcessor.from_pretrained(config.model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(config.model_name, device_map="auto")
            self.mode = "visual"
            self.prompt_format = llava_visual_format
            self.num_visual_tokens = int((self.model.config.vision_config.image_size / self.model.config.vision_config.patch_size) ** 2)
        elif "vicuna" in config.model_name.lower():
            self.processor = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
            self.model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
            self.mode = "text"
            self.prompt_format = text_format
        elif "qwen" in config.model_name.lower():
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="cuda", trust_remote_code=True).eval()
            self.mode ="visual"
            
        # accelerator = Accelerator()
        # accelerator.prepare_model(self.model, evaluation_mode=True)
        self.model.eval()
        self.model.to(self.device)
        
        
    def remode(self, mode="visual"):
        self.mode = mode
        if "llava" in self.config.model_name.lower():
            if mode == "visual":
                self.prompt_format = llava_visual_format
            else:
                self.prompt_format = text_format
    
    @torch.inference_mode()
    def chat(self, **kwargs):
        if self.mode == "text":
            text = kwargs["text"]
            
            messages = [[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]] * self.config.sampling_times
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)

                        
            # sampled_text = [self.prompt_format.format(system_prompt=self.prompt, user_input=text)] * self.config.sampling_times
            # inputs = self.processor(text=sampled_text, return_tensors="pt")
            # for k, v in inputs.items():
            #     if v is not None:
            #         inputs[k] = v.to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                **self.generate_config,
            )
            slen = inputs.input_ids.size(1)
            if "blip" in self.config.model_name.lower():
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            else:
                generated_text = self.processor.batch_decode(outputs[:, slen:], skip_special_tokens=True)[0].strip()
            
            is_scored = kwargs.get("is_scored")
            if is_scored:
                if "blip" not in self.config.model_name.lower() and "llava" not in self.config.model_name.lower() and "vicuna" in self.config.model_name.lower():
                    target_token_ids = torch.tensor(self.processor.convert_tokens_to_ids(["A", "B", "C", "D"]))
                else:    
                    target_token_ids = torch.tensor(self.processor.tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"]))
                probs = self.model(**inputs).logits[:, -1, :].detach().cpu()
                # probs = torch.nn.functional.softmax(self.model(**inputs).logits[:, -1, :].detach().cpu())
                target_probs = torch.index_select(probs, 1, target_token_ids).squeeze()
                # target_probs = torch.nn.functional.softmax(torch.index_select(probs, 1, target_token_ids)).squeeze()
                
        elif self.mode == "visual":
            text = kwargs.get("text")
            image = kwargs["image"]
            # if text is None:
            #     sampled_text = [self.prompt_format.format(system_prompt=self.prompt, user_input="")] * self.config.sampling_times
            # else:
            #     sampled_text = [self.prompt_format.format(system_prompt=self.prompt, user_input=text)] * self.config.sampling_times
            #     # sampled_text = [f"<|im_start|>system\n{self.prompt}<|im_end|><|im_start|>user\n<image>\n{text}<|im_end|><|im_start|>assistant\n"] * self.config.sampling_times
            # sampled_images = [image] * self.config.sampling_times
            
            if text is None:
                messages = [[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": self.prompt},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": ""},
                        ],
                    }
                ]] * self.config.sampling_times
            else:
                messages = [[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": self.prompt},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": text},
                        ],
                    }
                ]] * self.config.sampling_times
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            if "qwen" in self.config.model_name.lower():
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
            else:
                inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device)
            
            # inputs = self.processor(images=sampled_images, text=sampled_text, return_tensors="pt").to(self.model.device)
            # inputs = self.processor(images=sampled_images, text=sampled_text, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                **self.generate_config,
            )
            slen = inputs.input_ids.size(1)
            if "blip" in self.config.model_name.lower():
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            else:
                generated_text = self.processor.batch_decode(outputs[:, slen:], skip_special_tokens=True)[0].strip()
            
            is_scored = kwargs.get("is_scored")
            if is_scored:
                target_token_ids = torch.tensor(self.processor.tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"]))
                probs = self.model(**inputs).logits[:, -1, :].detach().cpu()
                # probs = torch.nn.functional.softmax(self.model(**inputs).logits[:, -1, :].detach().cpu())
                target_probs = torch.index_select(probs, 1, target_token_ids).squeeze()
                # target_probs = torch.nn.functional.softmax(torch.index_select(probs, 1, target_token_ids)).squeeze()
        
        elif self.mode == "zero_padding":
            text = kwargs["text"]
            sampled_text = [self.prompt_format.format(system_prompt=self.prompt, user_input=text)] * self.config.sampling_times
            inputs = self.processor(sampled_text, return_tensors="pt")
            inputs.input_ids = torch.cat([inputs.input_ids[: ,0].unsqueeze(0), self.model.pad_token_id * torch.ones((inputs.input_ids.size(0), self.num_visual_tokens)), inputs.input_ids[: , 1:]], dim=1)
            inputs.attention_mask = torch.cat([torch.ones((inputs.input_ids.size(0), self.num_visual_tokens)), inputs.attention_mask], dim=1)
            # position_ids = torch.tensor(list(range(576, 576 + inputs.input_ids.size(1)))).unsqueeze(0)
            # inputs.update({"position_ids": position_ids})
            for k, v in inputs.items():
                if v is not None:
                    inputs[k] = v.to(self.device)
            outputs = self.model.generate(
                **inputs,
                **self.generate_config,
            )
            slen = inputs.input_ids.size(1)
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            # print(generated_text)
            # input()
        
        is_scored = kwargs.get("is_scored")
        # print(generated_text)
        # input()
        if is_scored:
            return generated_text, target_probs.tolist()
        return generated_text