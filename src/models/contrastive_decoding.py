import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM

llava_visual_format = """"{system_prompt}USER: <image>\n{user_input}\nASSISTANT:"""
text_format = """"{system_prompt}USER: {user_input}\nASSISTANT:"""

class ContrastiveDecodingModel():
    def __init__(self, config, prompt="") -> None:
        # setup configure
        self.is_local = True
        self.config = config
        self.prompt = prompt + "\n"
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
        elif "llava" in config.model_name.lower():
            self.processor = LlavaProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
            self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
            self.mode = "visual"
            self.prompt_format = llava_visual_format
        self.model.to(self.device)
    
    @torch.inference_mode()
    def generate(self,
            text_for_text_logit,
            text_for_visual_logit,
            image,
            is_scored = False,
        ):
        # average_kl = 0
        max_kl = 0
        has_unfinished_sequence = True
        unfinished_sequences = torch.ones(self.config.sampling_times, dtype=torch.long, device=self.device)
        
        sampled_text_for_text_logit = [self.prompt_format.format(system_prompt=self.prompt, user_input=text_for_text_logit)] * self.config.sampling_times
        text_inputs = self.processor(text=sampled_text_for_text_logit, return_tensors="pt")
        # text_slen = text_inputs.input_ids.size(1)
        for k, v in text_inputs.items():
            if v is not None:
                text_inputs[k] = v.to(self.device)
        
        sampled_text_for_visual_logit = [self.prompt_format.format(system_prompt=self.prompt, user_input=text_for_visual_logit)] * self.config.sampling_times
        sampled_images = [image] * self.config.sampling_times
        visual_inputs = self.processor(images=sampled_images, text=sampled_text_for_visual_logit, return_tensors="pt")
        # visual_slen = visual_inputs.input_ids.size(1)
        for k, v in visual_inputs.items():
            if v is not None:
                visual_inputs[k] = v.to(self.device)
        
        generated_sequences = []
        # generated_text_sequences = []
        # generated_visual_sequences = []
        generated_length = 0
        while has_unfinished_sequence and generated_length < self.config.max_new_tokens:
            # calculate logits
            next_text_logits = self.model(**text_inputs).logits[:, -1, :]
            next_visual_logits = self.model(**visual_inputs).logits[:, -1, :]
            
            if self.config.kl > 0:
                kl_loss = (torch.nn.functional.kl_div(torch.nn.functional.softmax(next_text_logits, dim=-1), torch.nn.functional.softmax(next_visual_logits, dim=-1)).cpu().item() + torch.nn.functional.kl_div(torch.nn.functional.softmax(next_visual_logits, dim=-1), torch.nn.functional.softmax(next_text_logits, dim=-1)).cpu().item()) / 2
                # print(next_text_logits)
                # print(next_visual_logits)
                # print(kl_loss)
                # input()
                # if kl_loss > self.config.kl:
                if kl_loss > self.config.beta * max_kl:
                    next_token_logits = next_visual_logits - next_text_logits
                else:
                    next_token_logits = next_text_logits
                # average_kl = (generated_length * average_kl + kl_loss) / (generated_length + 1)
                max_kl = max(max_kl, kl_loss)
            else:
                if self.config.alpha > 0:
                    next_token_logits = (1 + self.config.alpha) * next_visual_logits - self.config.alpha * next_text_logits
                else:
                    next_token_logits = next_visual_logits - next_text_logits
            
            if is_scored and generated_length == 0:
                target_token_ids = torch.tensor(self.processor.tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"]))
                probs = next_token_logits.detach().cpu()
                target_probs = torch.index_select(probs, 1, target_token_ids).squeeze()
                
            # calculate scores
            next_tokens_scores = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # next_text_tokens_scores = torch.nn.functional.softmax(next_text_logits, dim=-1)
            # next_visual_tokens_scores = torch.nn.functional.softmax(next_visual_logits, dim=-1)
            
            # softmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            # next_text_tokens = torch.argmax(next_text_tokens_scores, dim=-1)
            # next_visual_tokens = torch.argmax(next_visual_tokens_scores, dim=-1)
            # print(next_tokens)
            # print(next_text_tokens)
            # print(next_visual_tokens)
            
            
            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + self.model.config.text_config.pad_token_id * (1 - unfinished_sequences)
            generated_sequences.append(next_tokens.detach().cpu())
            # generated_text_sequences.append(next_text_tokens.detach().cpu())
            # generated_visual_sequences.append(next_visual_tokens.detach().cpu())
            
            # update inputs
            text_inputs["input_ids"] = torch.cat([text_inputs["input_ids"], next_tokens[:, None]], dim=-1)
            visual_inputs["input_ids"] = torch.cat([visual_inputs["input_ids"], next_tokens[:, None]], dim=-1)
            text_inputs["attention_mask"] = torch.cat([text_inputs["attention_mask"], torch.ones_like(next_tokens[:, None])], dim=-1)
            visual_inputs["attention_mask"] = torch.cat([visual_inputs["attention_mask"], torch.ones_like(next_tokens[:, None])], dim=-1)
            
            # print(next_text_tokens)
            # print(next_visual_tokens)
            # print(text_inputs["input_ids"])
            # print(visual_inputs["input_ids"])
            # input()
            
            # decide is finished
            is_done = torch.isin(next_tokens, torch.tensor([self.model.config.text_config.eos_token_id]).to(self.device))
            unfinished_sequences = unfinished_sequences & ~is_done
            has_unfinished_sequence = unfinished_sequences.max() == 1
            
            # update length
            generated_length += 1

        generated_sequences = torch.cat(generated_sequences, dim=-1).unsqueeze(0)
        # print(generated_sequences)
        # print(generated_length)
        generated_text = self.processor.batch_decode(generated_sequences, skip_special_tokens=True)
        # generated_text_sequences = torch.cat(generated_text_sequences, dim=-1)
        # generated_visual_sequences = torch.cat(generated_visual_sequences, dim=-1)
        print(generated_text)
        # print(self.processor.batch_decode(generated_text_sequences, skip_special_tokens=True))
        # print(self.processor.batch_decode(generated_visual_sequences, skip_special_tokens=True))
        # input()
        if is_scored:
            return generated_text[0], target_probs.tolist()
        return generated_text