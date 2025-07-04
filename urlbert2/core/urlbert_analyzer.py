# /home/kong/urlbert/url_bert/urlbert2/core/url_analyzer.py
import torch
import torch.nn.functional as F
import requests
import random
import numpy as np

from pytorch_pretrained_bert import BertTokenizer 
from lime.lime_text import LimeTextExplainer

# config.py에서 필요한 모든 설정 값들을 임포트합니다.
from config import (
    PAD_SIZE, DEVICE, CLASS_LABELS, IMPORTANT_HEADERS,
    REQUEST_TIMEOUT_SECONDS, LIME_NUM_FEATURES, LIME_NUM_SAMPLES
)

# --- HTTP 헤더 정보 추출 함수 (원본 url.py와 동일) ---
def get_header_info(url: str) -> str:
    """
    주어진 URL에서 중요한 HTTP 헤더 정보를 추출합니다.
    URL 접근 실패 시 "NOHEADER"를 반환합니다.
    """
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/114 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 Version/14.0 Mobile/15E148 Safari/604.1"
    ]
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }

    try:
        # timeout 설정에 config의 REQUEST_TIMEOUT_SECONDS 사용
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, allow_redirects=True)
        resp_headers = response.headers

        important = {
            k: resp_headers.get(k, "") for k in IMPORTANT_HEADERS
        }

        header_str = ", ".join(f"{k}: {v}" for k, v in important.items() if v)
        return header_str if header_str else "NOHEADER"

    except requests.exceptions.RequestException:
        return "NOHEADER"
    except Exception: # 기타 예외 처리
        return "NOHEADER"

# --- 데이터 전처리 함수 (원본 url.py와 동일하게 BertTokenizer 타입 힌트 유지) ---
def preprocess_url_for_inference(url: str, header_info: str, tokenizer: BertTokenizer, pad_size: int = PAD_SIZE):
    """
    URL과 추출된 헤더 정보를 BERT 모델 입력 형식으로 전처리합니다.
    """
    text = f"{url} [SEP] {header_info}"
    tokenized_text = tokenizer.tokenize(text)
    
    tokens = ["[CLS]"] + tokenized_text + ["[SEP]"]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    types = [0] * (len(ids)) 
    masks = [1] * len(ids)

    if len(ids) < pad_size:
        types = types + [1] * (pad_size - len(ids)) 
        masks = masks + [0] * (pad_size - len(ids))
        ids = ids + [0] * (pad_size - len(ids))
    else:
        types = types[:pad_size]
        masks = masks[:pad_size]
        ids = ids[:pad_size]

    assert len(ids) == len(masks) == len(types) == pad_size

    return (
        torch.tensor([ids], dtype=torch.long).to(DEVICE),
        torch.tensor([types], dtype=torch.long).to(DEVICE),
        torch.tensor([masks], dtype=torch.long).to(DEVICE)
    )

# --- LIME 설명자를 위한 예측 함수 ---
def lime_predictor_fn(texts, model, tokenizer): 
    probabilities = []
    
    for text_input in texts:
        parts = text_input.split(" [SEP] ", 1)
        url_part = parts[0]
        header_part = parts[1] if len(parts) > 1 else "NOHEADER"

        input_ids, input_types, input_masks = preprocess_url_for_inference(
            url_part, header_part, tokenizer, PAD_SIZE
        )

        with torch.no_grad():
            outputs = model([input_ids, input_types, input_masks])
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            probabilities.append(probs[0])

    return np.array(probabilities)

# --- URL 분류 및 설명 생성 함수 (메인 추론 함수) ---
def classify_url_with_explanation(url: str, model, tokenizer) -> dict: 
    print(f"\nURL 분석 시작: {url}")
    header_info = get_header_info(url)
    print(f"추출된 헤더 정보: {header_info if header_info != 'NOHEADER' else '없음'}")

    input_ids, input_types, input_masks = preprocess_url_for_inference(
        url, header_info, tokenizer, PAD_SIZE
    )

    with torch.no_grad():
        outputs = model([input_ids, input_types, input_masks])
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_id = torch.argmax(probabilities, dim=1).item()

    predicted_label = CLASS_LABELS[predicted_class_id]
    confidence = probabilities[0][predicted_class_id].item() * 100

    full_text_for_lime = f"{url} [SEP] {header_info}"

    try:
        explainer = LimeTextExplainer(class_names=list(CLASS_LABELS.values()))

        explanation = explainer.explain_instance(
            full_text_for_lime,
            classifier_fn=lambda texts: lime_predictor_fn(texts, model, tokenizer),
            labels=[0, 1], 
            num_features=LIME_NUM_FEATURES, # config에서 임포트한 변수 사용
            num_samples=LIME_NUM_SAMPLES # config에서 임포트한 변수 사용
        )

        explanation_list = explanation.as_list(label=predicted_class_id)
        
        reason_phrases = []
        positive_influences = [word for word, weight in explanation_list if weight > 0]
        negative_influences = [word for word, weight in explanation_list if weight < 0]

        if predicted_label == 'malicious':
            if positive_influences:
                reason_phrases.append(f"특히 '{', '.join(positive_influences[:3])}' 등의 특징이 악성으로 의심됩니다.")
            else:
                reason_phrases.append("주요 악성 징후를 명확히 파악하기 어렵지만, 전반적인 패턴이 악성으로 분류되었습니다.")
        else: # 'benign'
            if negative_influences: 
                reason_phrases.append(f"이 URL은 '{', '.join(negative_influences[:3])}' 등의 일반적인 패턴을 가지고 있습니다.")
            else:
                reason_phrases.append("전반적으로 안전한 URL 패턴을 보입니다.")

        reason_summary = "이 URL은 " + predicted_label.upper() + "로 판단됩니다. " + " ".join(reason_phrases)

        print("\n--- 상세 설명 (LIME 분석) ---")
        for word, weight in explanation_list:
            print(f"  - '{word}': {weight:.4f} (영향도)")
        print("----------------------------")

    except Exception as e:
        print(f"LIME 설명 생성 중 오류 발생: {e}")
        reason_summary = f"이 URL은 {predicted_label.upper()}로 판단됩니다 (확신도: {confidence:.2f}%). 판단 근거를 설명하는 중 오류가 발생했습니다."
        explanation_list = []

    print(f"분류 결과: **{predicted_label.upper()}** (확신도: {confidence:.2f}%)")
    print(f"설명 요약: {reason_summary}")

    return {
        "predicted_label": predicted_label,
        "confidence": f"{confidence:.2f}%",
        "reason_summary": reason_summary,
        "detailed_explanation": explanation_list
    }