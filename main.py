import argparse
import json
import yaml
import os

from builders.model_builder import build_model
from utils.postprocess import postprocess
from utils.metrics import run_evaluation
from huggingface_hub import login


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    return parser.parse_args()


def load_done_ids(path):
    done_ids = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue               # bỏ qua dòng bị corrupt
    return done_ids


def load_results(path):
    results = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def main():
    args = parse_args()

    # Load config
    with open(args.config_file, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load model
    model = build_model(cfg)

    # Load data
    with open(cfg['data']['input'], encoding='utf-8') as f:
        data = json.load(f)

    # Prepare output
    output_path = cfg['data']['output']
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Resume
    done_ids = load_done_ids(output_path)

    # Inference loop 

    with open(output_path, 'a', encoding='utf-8') as f:
        for i, sample in enumerate(data):
            if i in done_ids:
                continue

            text = sample['text']
            prompt = model.build_prompt(text)
            raw = model.generate(prompt)
            para = postprocess(raw, text)

            item = {
                "id": i,
                "original": text,
                "raw_output": raw,
                "paraphrase": para
            }

            f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"[{i+1}/{len(data)}] Done")

    # Load lại toàn bộ kết quả để evaluate
    results = load_results(output_path)

    # Evaluate
    run_evaluation(results)


if __name__ == '__main__':
    main()