import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseLLM
from builders.registry import register_model
from utils.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


@register_model("qwen")
class QwenLLM(BaseLLM):
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
        self.model.eval()

    def build_prompt(self, text):
        user_content = USER_PROMPT_TEMPLATE.format(text=text)
        # Qwen2.5 dùng apply_chat_template thay vì tự build string
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True   # tự thêm <|im_start|>assistant\n
        )

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

        # Chỉ decode phần model sinh ra, bỏ phần prompt
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()