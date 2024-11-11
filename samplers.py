import os
import torch
from typing import Any

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


from eval_types import MessageList, SamplerBase

class GemmaCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API for o1 models
    """

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
                                                            attn_implementation="flash_attention_2",
                                                            ) 
            
        else:
            raise ValueError(f"Unsupported dtype: {d_type}")
        
        

    def __call__(self, message_list: MessageList) -> str:

        input_ids = self.tokenizer.apply_chat_template(message_list, return_tensors="pt", return_dict=True).to(self.device)
        response = self.model.generate(**input_ids, max_new_tokens=self.max_tokens)
        response = self.tokenizer.decode(response[0])

        return response
    


class LlamaCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API for o1 models
    """

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

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=d_type,
            device_map=self.device,
            model_kwargs={"attn_implementation": "flash_attention_2",}
        )
        

    def __call__(self, message_list: MessageList) -> str:

        response = self.pipe(
            message_list,
            max_new_tokens=self.max_tokens,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )

        return response[0]["generated_text"][-1]["content"]
    
#Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering. If today is Monday, then what day is tomorrow A) Thursday B) Wednesday C) Tuesday D) Monday