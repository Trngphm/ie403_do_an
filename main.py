import argparse, json, yaml
from builders.model_builder import build_model
from utils.postprocess import postprocess
from utils.metrics import run_evaluation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    return parser.parse_args()



def main():
    args = parse_args()
    with open(args.config_file, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Load model
    model = build_model(cfg)

    # Load data
    with open(cfg['data']['input'], encoding='utf-8') as f:
        data = json.load(f)

    # Paraphrase
    pp_cfg = cfg.get('postprocess', {})
    results = []
    for i, sample in enumerate(data):
        text = sample['text']
        prompt = model.build_prompt(text)
        raw = model.generate(prompt)
        para = postprocess(raw, text)
        results.append({"original": text, "paraphrase": para})
        print(f"Done {i+1}/{len(data)}")

    # Save
    with open(cfg['data']['output'], 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Evaluate
    run_evaluation(results)

if __name__ == '__main__':
    main()