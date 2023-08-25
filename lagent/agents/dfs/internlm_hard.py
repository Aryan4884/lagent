#!/usr/bin/env python
# coding=utf-8
import time
from termcolor import colored
from typing import Optional, List
import torch
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from lagent.agents.dfs.utils import process_system_message, react_parser
from lagent.agents.dfs.llama_utils import generate_stream

class InternLMHard:
    def __init__(self, model_name_or_path: str, template:str="tool-llama-single-round", device: str="cuda", cpu_offloading: bool=False) -> None:
        super().__init__()
        self.model_name = model_name_or_path
        self.template = template
        self.tokenizer = AutoTokenizer.from_pretrained('pretrain_models/internlm_v1_1', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.use_gpu = (True if device == "cuda" else False)
        if (device == "cuda" and not cpu_offloading) or device == "mps":
            self.model.to(device)

    def prediction(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        with torch.no_grad():
            gen_params = {
                "model": "",
                "prompt": prompt,
                "temperature": 0.5,
                "max_new_tokens": 512,
                "stop": "</s>",
                "stop_token_ids": None,
                "echo": False
            }
            generate_stream_func = generate_stream
            output_stream = generate_stream_func(self.model, self.tokenizer, gen_params, "cuda", 2048, force_generate=True)
            outputs = self.return_output(output_stream)
            prediction = outputs.strip()
        return prediction

    def return_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                pre = now
        return " ".join(output_text)

    def parse(self, _history, functions, process_id):
        prompt = ''
        roles = {"system": ['<|System|>:', '<TOKENS_UNUSED_2>\n'], 
            "user": ['<|User|>:', '<eoh>\n'], 
            "function": ['<|System|>:', '<TOKENS_UNUSED_2>\n'], 
            "assistant": ['<|Bot|>:', '<eoa>\n']}
        
        for message in _history:
            start, end = roles[message['role']]
            content = message['content']
            if message['role'] == "system" and functions != []:
                content = process_system_message(content, functions)
            prompt += f"{start}{content}{end}"
        prompt += "<|Bot|>:"
        
        import time
        start = time.time()
            
        if functions != []:
            predictions = self.prediction(prompt)
        else:
            predictions = self.prediction(prompt)

        end = time.time()
        print("===== %.2f s ====" % (end - start))

        decoded_token_len = len(self.tokenizer(predictions))
        if process_id == 0:
            print(f"[process({process_id})]total tokens: {decoded_token_len}")

        # react format prediction
        thought, action, action_input = react_parser(predictions)
        message = {
            "role": "assistant",
            "content": thought,
            "function_call": {
                "name": action,
                "arguments": action_input
            }
        }
        return message, 0, decoded_token_len