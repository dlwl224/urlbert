import torch
import torch.nn.functional as F
import requests
import random
import numpy as np

from pytorch_pretrained_bert import BertTokenizer  # 기존 임포트 유지

# config.py에서 필요한 모든 설정 값들을 임포트합니다.
from config import (
    PAD_SIZE, DEVICE, CLASS_LABELS, IMPORTANT_HEADERS,
    REQUEST_TIMEOUT_SECONDS
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

# --- URL 분류 함수 (LIME 로직 없음) ---
# 함수 이름을 classify_url로 변경하여 LIME 기능이 없음을 명확히 합니다.
def classify_url(url: str, model, tokenizer) -> dict: 
    """
    주어진 URL을 BERT 모델을 사용하여 악성/정상으로 분류하고 확신도를 반환합니다.
    LIME 설명 로직은 포함되지 않습니다.
    """
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

    print(f"분류 결과: **{predicted_label.upper()}** (확신도: {confidence:.2f}%)")

    # LIME 관련 필드 제거 또는 빈 값으로 반환
    return {
        "predicted_label": predicted_label,
        "confidence": f"{confidence:.2f}%",
        "reason_summary": "LIME 설명 로직이 포함되지 않은 버전입니다.", # LIME 설명 없음
        "detailed_explanation": [] # LIME 상세 설명 없음
    }

# --- 사용 예시 (이 부분은 app.py에서 호출되므로, 여기서는 제거하거나 간단히만 둡니다) ---
# if __name__ == "__main__":
#     # 이 파일은 주로 다른 모듈에서 임포트하여 사용됩니다.
#     # 직접 실행 시 테스트 코드를 여기에 넣을 수 있습니다.
#     print("url_analyzer.py는 주로 다른 모듈에 의해 임포트되어 사용됩니다.")
#     print("테스트를 원하시면 app.py를 실행해주세요.")