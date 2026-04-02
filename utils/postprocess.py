import re

def postprocess(text, original):
    if not text:
        return ""

    # 1. Xóa ký tự thừa
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # nhiều space → 1 space

    # 2. Xóa lặp câu đơn giản
    sentences = text.split(". ")
    seen = set()
    filtered = []
    for s in sentences:
        if s not in seen:
            filtered.append(s)
            seen.add(s)
    text = ". ".join(filtered)

    # 3. Giới hạn độ dài (<= +10 từ)
    orig_len = len(original.split())
    max_len = orig_len + 10
    words = text.split()

    if len(words) > max_len:
        text = " ".join(words[:max_len])

    # 5. Xóa nếu quá ngắn (fail)
    if len(text.split()) < 5:
        return ""

    return text.strip()