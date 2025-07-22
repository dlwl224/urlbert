import random
import numpy as np
import torch

from config import SEED 
from core.model_loader import load_inference_model 
from core.urlbert_analyzer import classify_url_and_explain

# --- 재현성을 위한 시드 고정  ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    # 모델 및 토크나이저 로드 (한 번만 로드)
    model, tokenizer = load_inference_model()

    print("\n--- URL 분석 서비스 시작 (테스트용) ---")

    test_urls = [
        "https://www.google.com",
        "https://www.naver.com",
        "https://flowto.it/8noX2wTHr?fc=0",
        "https://ezdriverma.com-xaaawn.vip/",
        "https://com-pif.xin/",
        "https://www.c.cdnhwc6.com",
        "http://thisurldoesnotexist12345.com",
        "http://192.168.1.99:8080/nonexistent",
        "http://phishingsite.com/login?userid=123",
        "https://securebank.co.kr/main",
        "http://malicious.example.com/download/malware.exe",
        "https://short.ly/malware"
    ]

    for url in test_urls:
        print(f"\n--- URL 분석 시작: {url} ---")
        
        result = classify_url_and_explain(url, model, tokenizer)

        print("\n--- 분석 결과 요약 ---")
        print(f"URL: {url}")
        
        try:
            confidence_value = float(result['confidence'])

            # 여기서 predicted_label 키 대신 is_malicious로 판단하여 라벨 생성
            predicted_label = "MALICIOUS" if result['is_malicious'] else "BENIGN"
            print(f"분류: {predicted_label} (확신도: {confidence_value:.2f}%)")

        except (ValueError, TypeError, KeyError):
            print("⚠️ 확신도 값이 유효하지 않거나 'confidence' 키가 없습니다.")
        
        print(f"전체 설명 요약: {result.get('reason_summary', '요약 없음')}")

        if result.get('detailed_explanation'):
            print(result['detailed_explanation'])
        else:
            print("\n상세 LIME 설명이 생성되지 않았습니다.")
        
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
