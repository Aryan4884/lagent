from typing import Dict, List, Optional

import torch

from .base_llm import BaseModel
from .huggingface import HFTransformerCasualLM

class ToolLLaMA(HFTransformerCasualLM):

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path,
            use_fast=False,
            model_max_length=8192,  # NOTE: can be passed throught tokenizer_kwargs
            **tokenizer_kwargs)
        # if self.tokenizer.pad_token_id is None:
        #     self.tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
            # self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self, path: str, model_kwargs: dict):
        from transformers import AutoModelForCausalLM
        model_kwargs.setdefault('torch_dtype', torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, low_cpu_mem_usage=True, **model_kwargs)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def generate_from_template(self, templates, max_out_len: int, **kwargs):
        """Generate completion from a list of templates.

        Args:
            templates (List[PromptType]): A list of templates.
            max_out_len (int): The maximum length of the output.
        """
        inputs = self.parse_template(templates)
        inputs = inputs[:-1] + '\n'
        import time
        start = time.time()
        response = self.generate(inputs, max_out_len=max_out_len, **kwargs)
        end = time.time()
        print("===== %.2f s ====" % (end - start))
        return response.replace(
            self.template_parser.roles['assistant']['end'].strip(),
            '').strip()

    @torch.inference_mode()
    def generate(self, inputs: List[str], max_out_len: int,
                 **kwargs) -> List[str]:
        if isinstance(inputs, str):
            inputs = [inputs]
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]
        input_ids = self.tokenizer(
            inputs, truncation=True,
            max_length=self.max_seq_len - max_out_len)['input_ids']
        input_ids = torch.tensor(input_ids, device=self.model.device)
        # kwargs['top_k'] = 32000; kwargs['top_p'] = 1.0
        # kwargs['num_beams'] = 1; kwargs['do_sample'] = False
        outputs = self.model.generate(
            input_ids, max_new_tokens=max_out_len, **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, input_ids.shape[1]:]

        decodeds = [self.tokenizer.decode(output, skip_special_tokens=True, 
            spaces_between_special_tokens=False) for output in outputs]
        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        return decodeds[0]