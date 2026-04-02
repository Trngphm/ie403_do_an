# ===============================
# 1. Cài đặt thư viện cần thiết
# ===============================
# pip install torch transformers sacrebleu bert-score underthesea python-Levenshtein

import torch
from underthesea import word_tokenize
from nltk.translate.meteor_score import single_meteor_score
import sacrebleu
from bert_score import score
import Levenshtein

# ===============================
# 3. Hàm tính các metric
# ===============================
def evaluate_metrics_vi(original, paraphrase):
    metrics = {}

    # --- Tokenize tiếng Việt bằng underthesea ---
    original_tokens = word_tokenize(original, format="text").split()
    paraphrase_tokens = word_tokenize(paraphrase, format="text").split()

    # --- METEOR ---
    metrics['METEOR'] = single_meteor_score(original_tokens, paraphrase_tokens)

    # --- BLEU (sacrebleu) ---
    bleu = sacrebleu.corpus_bleu([paraphrase], [[original]])
    metrics['BLEU'] = bleu.score

    # --- Levenshtein Distance ---
    metrics['LD'] = Levenshtein.distance(original, paraphrase)

    return metrics

# ===============================
# 4. BERTScore (batch)
# ===============================
def run_evaluation(output):
    originals = [d['original'] for d in output]
    paraphrases = [d['paraphrase'] for d in output]

    P, R, F1 = score(paraphrases, originals, lang="vi", verbose=False)
    bertscores = F1.tolist()  # F1 tensor → list float

    # ===============================
    # 5. Tính metrics cho từng cặp và lưu lại để tính trung bình
    # ===============================
    all_metrics = []

    for i, d in enumerate(output):
        metrics = evaluate_metrics_vi(d['original'], d['paraphrase'])
        metrics['BERTScore-F1'] = bertscores[i]
        all_metrics.append(metrics)  # lưu lại để tính trung bình
        
        print(f"Cặp {i+1}:")
        print(f"  METEOR      : {metrics['METEOR']:.4f}")
        print(f"  BLEU        : {metrics['BLEU']:.2f}")
        print(f"  LD          : {metrics['LD']}")
        print(f"  BERTScore-F1: {metrics['BERTScore-F1']:.4f}")
        print("-"*50)

    # ===============================
    # 6. Tính trung bình trên toàn bộ cặp
    # ===============================
    avg_metrics = {}
    for key in ['METEOR', 'BLEU', 'LD', 'BERTScore-F1']:
        avg_metrics[key] = sum(d[key] for d in all_metrics) / len(all_metrics)

    print("=== Trung bình các metric trên toàn bộ cặp ===")
    for k, v in avg_metrics.items():
        if k in ['METEOR', 'BERTScore-F1']:
            print(f"{k:12}: {v:.4f}")
        elif k == 'BLEU':
            print(f"{k:12}: {v:.2f}")
        else:
            print(f"{k:12}: {v:.0f}")
