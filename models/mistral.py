import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseLLM
from builders.registry import register_model
from utils.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


@register_model("mistral")
class MistralLLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)
        self.load_model()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        
    def build_prompt(self, text):
        user_content = USER_PROMPT_TEMPLATE.format(text=text)
        return f"[INST] {SYSTEM_PROMPT}\n\n{user_content} [/INST]"

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            min_new_tokens=5,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            top_p=self.config.top_p,
            top_k=self.config.top_k
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return text.split('[/INST]')[-1].strip()