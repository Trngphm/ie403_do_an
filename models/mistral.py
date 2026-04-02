import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseLLM
from builders.registry import register_model
from utils.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


@register_model("mistral")
class MistralLLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)
        self.load_model()

    def load_model(self):
        m_cfg = self.config["model"]
        self.model = AutoModelForCausalLM.from_pretrained(
            m_cfg['pretrained'],
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(m_cfg['pretrained'])

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        
    def build_prompt(self, text):
        user_content = USER_PROMPT_TEMPLATE.format(text=text)
        return f"[INST] {SYSTEM_PROMPT}\n\n{user_content} [/INST]"

    def generate(self, prompt):
        gen_cfg = self.config['generation']
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            min_new_tokens=gen_cfg.get('min_new_tokens', 5),
            max_new_tokens=gen_cfg.get('max_new_tokens', 256),
            do_sample=gen_cfg.get('do_sample', True),
            top_p=gen_cfg.get('top_p', 0.95),
            top_k=gen_cfg.get('top_k', 50),
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return text.split('[/INST]')[-1].strip()