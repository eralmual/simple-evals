import os
import torch
from typing import Any

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline


from eval_types import MessageList, SamplerBase

class ModelCompletionSampler(SamplerBase):

    def __init__(
        self,
        model_id: str = "google/gemma-2-2b-it",
        device: str = "cpu",
        max_tokens: int = 1024,
        d_type = torch.bfloat16,
    ):
        self.model_id = model_id
        self.image_format = "url"
        self.device = device
        self.max_tokens = max_tokens

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Load quantized or base model
        self.model = None
        if(d_type == torch.bfloat16):
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, 
                                                            device_map=self.device,
                                                            torch_dtype=d_type,
                                                            ) 
        elif(d_type == 4):
            print("Running ", self.model_id, " with 4 bit quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=self.device,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
            )

            
        else:
            raise ValueError(f"Unsupported dtype: {d_type}")
        
        

    def __call__(self, message_list: MessageList) -> str:

        input_ids = self.tokenizer.apply_chat_template(message_list, return_tensors="pt", return_dict=True).to(self.device)
        response = self.model.generate(**input_ids, max_new_tokens=self.max_tokens)
        response = self.tokenizer.decode(response[0])

        return response
    


class PipelineCompletionSampler(SamplerBase):

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        max_tokens: int = 1024,
        d_type = torch.bfloat16,
        device: str = "cpu",
    ):
        self.max_tokens = max_tokens
        self.model_id = model_id
        self.device = device

        if(d_type == 4):
            print("Running", self.model_id, "with 4 bit quantization")
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                device_map=self.device,
                model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)}
            )
        else:
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=d_type,
                device_map=self.device,
            )
        

    def __call__(self, message_list: MessageList) -> str:

        response = self.pipe(
            message_list,
            max_new_tokens=self.max_tokens,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )

        return response[0]["generated_text"][-1]["content"]
    
