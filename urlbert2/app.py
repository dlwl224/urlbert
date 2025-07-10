# /home/kong/urlbert/url_bert/urlbert2/app.py
import random
import numpy as np
import torch

from config import SEED 

from core.model_loader import load_inference_model 
from core.urlbert_analyzer import classify_url_and_explain # predict_url, explain_prediction_with_lime은 classify_url_and_explain 내에서 호출되므로 여기서 직접 임포트할 필요는 없습니다.

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
        "https://flowto.it/8noX2wTHr?fc=0", # 난독화된 서브도메인, 특이 TLD
        "https://ezdriverma.com-xaaawn.vip/", # 서브도메인 난독화, 특이 TLD
        "https://com-pif.xin/", # 특이 TLD
        "https://www.c.cdnhwc6.com", # 짧은 c, 긴 숫자/문자 조합
        "http://thisurldoesnotexist12345.com", # 존재하지 않는 도메인, http
        "http://192.168.1.99:8080/nonexistent", # 사설 IP 주소, http
        "http://phishingsite.com/login?userid=123", # 피싱 의심, http
        "https://securebank.co.kr/main", # 정상 사이트 -> 이전 대화에서 이 URL이 MALICIOUS로 분류되었다고 하셨는데, 실제 모델이 그렇게 예측하는지 확인 필요
        "http://malicious.example.com/download/malware.exe", # 악성 예시
        "https://short.ly/malware" # 단축 URL 예시
    ]

    for url in test_urls:
        print(f"\n--- URL 분석 시작: {url} ---")
        
        # 예측 및 설명을 모두 수행하는 함수 호출
        result = classify_url_and_explain(url, model, tokenizer)
        
        print("\n--- 분석 결과 요약 ---")
        print(f"URL: {url}")
        
        # 이전 오류 수정: confidence가 문자열이므로 float으로 변환하여 포맷팅
        try:
            confidence_value = float(result['confidence'])
            print(f"분류: {result['predicted_label'].upper()} (확신도: {confidence_value:.2f}%)") # 확신도에 % 추가
        except ValueError:
            print(f"분류: {result['predicted_label'].upper()} (확신도: {result['confidence']})") # 변환 실패 시 원본 문자열 출력
            print("경고: 확신도 값이 숫자가 아닌 문자열이어서 정확한 포맷팅이 불가능합니다.")

        print(f"전체 설명 요약: {result['reason_summary']}")
        
        # 상세 LIME 설명 출력
        # result['detailed_explanation']은 이미 formatted_detailed_explanation (문자열)입니다.
        if result['detailed_explanation']:
            print(result['detailed_explanation']) # 문자열 전체를 출력합니다.
        else:
            print("\n상세 LIME 설명이 생성되지 않았습니다.")
            
        print("\n" + "=" * 80 + "\n") # 각 URL 분석 결과 구분선


if __name__ == "__main__":
    main()