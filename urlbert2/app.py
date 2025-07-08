import random
import numpy as np
import torch

# config 모듈에서 SEED를 정확하게 임포트합니다.
from config import SEED 

# core 모듈에서 필요한 함수 임포트 (모델과 토크나이저는 함수 반환값으로 받음)
from core.model_loader import load_inference_model 
# classify_url_with_explanation 대신 classify_url 함수를 임포트합니다.
from core.urlbert_analyzer import classify_url 

# --- 재현성을 위한 시드 고정 (원본 url.py와 동일하게 유지) ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    # 모델 및 토크나이저 로드 (load_inference_model()은 모델과 토크나이저 객체를 반환합니다)
    model, tokenizer = load_inference_model()

    print("\n--- URL 분석 서비스 시작 (테스트용) ---")

    # 예시 URL 리스트
    test_urls = [
        "https://www.google.com",
        "https://www.naver.com",
        "https://flowto.it/8noX2wTHr?fc=0",
        "https://ezdriverma.com-xaaawn.vip/",
        "https://com-pif.xin/",
        "https://www.c.cdnhwc6.com",
        "http://thisurldoesnotexist12345.com",
        "http://192.168.1.99:8080/nonexistent"
    ]

    for url in test_urls:
        # classify_url 함수에 로드된 model과 tokenizer 객체를 직접 전달
        # 이 함수 내부에서 이미 분석 과정이 출력됩니다.
        result = classify_url(url, model, tokenizer)
        
        # --- 분석 결과 요약 출력 (명시적으로 다시 추가) ---
        print("\n--- 분석 결과 요약 ---")
        print(f"URL: {url}")
        print(f"분류: {result['predicted_label'].upper()} (확신도: {result['confidence']})")
        print(f"설명: {result['reason_summary']}") # LIME 설명이 없으므로 고정 메시지 출력
        # print(f"자세한 기여도: {result['detailed_explanation']}") # LIME이 없으므로 주석 처리 유지
        print("-" * 70) # 각 URL 분석 결과 사이에 구분선 추가

if __name__ == "__main__":
    main()