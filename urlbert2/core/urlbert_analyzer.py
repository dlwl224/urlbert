# /home/kong/urlbert/url_bert/urlbert2/core/url_analyzer.py

import os
import sys

# 현재 파일의 디렉토리 (core)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉토리 (urlbert2)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 프로젝트 루트 디렉토리를 Python 검색 경로에 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import requests
import random
import numpy as np

from pytorch_pretrained_bert import BertTokenizer 
from lime.lime_text import LimeTextExplainer

from config import (
    PAD_SIZE, DEVICE, CLASS_LABELS, IMPORTANT_HEADERS,
    REQUEST_TIMEOUT_SECONDS, LIME_NUM_FEATURES, LIME_NUM_SAMPLES,
    TRUSTED_DOMAINS_FOR_EXPLANATION
)

# --- HTTP 헤더 정보 추출 함수 ---
def get_header_info(url: str) -> str:
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
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, allow_redirects=True)
        resp_headers = response.headers
        
        # IMPORTANT_HEADERS가 리스트로 정의되어 있다면
        important = {
            k: resp_headers.get(k, "") for k in IMPORTANT_HEADERS
        } 
        header_str = ", ".join(f"{k}: {v}" for k, v in important.items() if v)
        return header_str if header_str else "NOHEADER"
    except requests.exceptions.RequestException:
        return "NOHEADER"
    except Exception:
        return "NOHEADER"

# --- 데이터 전처리 함수  ---
def preprocess_url_for_inference(url: str, header_info: str, tokenizer: BertTokenizer, pad_size: int = PAD_SIZE):
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

# --- LIME 설명자를 위한 예측 함수  ---
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

# --- URL 분류 및 설명 생성 함수 ---
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
            num_features=LIME_NUM_FEATURES, 
            num_samples=LIME_NUM_SAMPLES 
        )

        explanation_list = explanation.as_list(label=predicted_class_id)
        
        # --- LIME 설명 요약 로직 개선 ---
        reason_phrases = []
        
        # 이해하기 쉬운 키워드를 필터링하고 설명 문구를 생성하는 헬퍼 함수
        # 반환값은 (설명 문구, 해당 특징의 일반적인 의미 설명) 튜플
        def get_understandable_explanation_text_and_meaning(word, weight, predicted_label_context_for_meaning):
            # 1단계 필터링: BERT 특수 토큰, 너무 긴 문자열, 불필요한 쿠키/세션 이름
            # word.strip().lower()를 사용하여 어떤 형태의 SEP도 확실히 걸러냄
            if word.strip().lower() in ["[sep]", "[cls]", "[pad]"] or \
               len(word) > 50 or \
               word.upper() in ["NID", "SID", "PHPSESSID", "AEC"]:
                return None, None

            # 2단계: 영향도에 따른 설명 문구와 의미 생성
            display_text = ""
            meaning = ""
            
            # 예측된 라벨에 대한 기여 방향
            # LIME의 weight는 해당 클래스로의 기여도.
            # 즉, 'benign'일 때 양수 weight는 benign으로 가는 데 긍정적, 음수 weight는 benign으로 가는 데 부정적(malicious로 유도)
            # 'malicious'일 때 양수 weight는 malicious로 가는 데 긍정적, 음수 weight는 malicious로 가는 데 부정적(benign으로 유도)
            
            # --- 신뢰할 수 있는 도메인 ---
            if word.lower() in TRUSTED_DOMAINS_FOR_EXPLANATION:
                meaning = "웹사이트의 주소(도메인)는 해당 웹사이트의 신뢰도를 나타내는 중요한 요소입니다. 널리 알려지고 안전하게 사용되는 도메인은 일반적으로 안전한 URL로 판단될 가능성을 높입니다."
                if predicted_label_context_for_meaning == "정상으로 판단될 가능성을" and weight > 0:
                    display_text = f"'{word}'와 같은 **널리 알려지고 신뢰할 수 있는 도메인**이 정상으로 판단될 가능성을 높였습니다."
                elif predicted_label_context_for_meaning == "악성으로 판단될 가능성을" and weight < 0: # 악성인데 음수 영향 -> 악성 판단을 감소 시킴 (즉, 정상쪽으로 미는 요소)
                    display_text = f"'{word}'와 같은 **신뢰할 수 있는 도메인**이 악성으로 판단될 가능성을 다소 감소시켰습니다. 이는 악성 URL이 정상 사이트를 모방하려 할 때 나타날 수 있는 패턴입니다."
                else: # 그 외의 경우 (예: 정상인데 음수 영향 - 이 경우는 '정상으로 판단될 가능성을 감소시켰다'가 더 명확)
                    display_text = f"'{word}'와 같은 **신뢰할 수 있는 도메인**이 {predicted_label_context_for_meaning} {('증가시키는' if weight > 0 else '감소시키는')} 영향을 미쳤습니다."
            
            # --- HTTPS/HTTP ---
            elif word.lower() == 'https':
                meaning = "HTTPS는 웹사이트와 사용자 간의 통신을 암호화하여 데이터를 안전하게 보호하는 프로토콜입니다. HTTPS 사용은 URL의 보안 수준을 높이는 긍정적인 신호입니다."
                display_text = f"안전한 **HTTPS 연결**이 {predicted_label_context_for_meaning} {('증가시키는' if weight > 0 else '감소시키는')} 영향을 미쳤습니다."
            elif word.lower() == 'http':
                meaning = "HTTP는 암호화되지 않은 통신 프로토콜로, 데이터가 노출될 위험이 있습니다. 최신 웹사이트는 대부분 HTTPS를 사용하므로, HTTP만 사용하는 경우 의심스러운 요소로 작용할 수 있습니다."
                display_text = f"보안에 취약한 'HTTP' 프로토콜이 {predicted_label_context_for_meaning} {('증가시키는' if weight > 0 else '감소시키는')} 영향을 미쳤습니다."

            # --- WWW ---
            elif word.lower() == 'www':
                meaning = "대부분의 일반적인 웹사이트는 'www' 접두사를 사용합니다. 이는 표준적인 웹 주소 형태로, URL의 정상성을 나타내는 신호로 작용할 수 있습니다."
                display_text = f"'WWW' 접두사 사용이 {predicted_label_context_for_meaning} {('증가시키는' if weight > 0 else '감소시키는')} 영향을 미쳤습니다."
            
            # --- NOHEADER --- 
            elif word.lower() == 'noheader':
                meaning = "URL 접속 시 **HTTP 헤더 정보가 전혀 없거나 비정상적인 경우**, 이는 서버 설정의 문제이거나, 정보를 숨겨 분석을 어렵게 하려는 악의적인 시도로 해석될 수 있습니다. 정상적인 웹사이트는 일반적으로 다양한 헤더 정보를 주고받습니다."
                display_text = f"**HTTP 헤더 정보 부재**가 {predicted_label_context_for_meaning} {('증가시키는' if weight > 0 else '감소시키는')} 영향을 미쳤습니다."
            
            # --- 중요 HTTP 헤더 이름 --- 
            elif word in IMPORTANT_HEADERS: 
                meaning = f"'{word}' 헤더는 웹 서버와 클라이언트 간의 통신 정보를 담고 있습니다. 이 헤더의 내용이나 존재 여부는 웹사이트의 특성(예: 사용된 웹 서버 종류, 콘텐츠 타입, 쿠키 설정 등)을 파악하는 데 중요합니다. 악성 사이트의 경우 정상적인 헤더가 없거나, 피싱을 위해 특정 헤더를 조작하는 등 비정상적인 값을 가질 수 있습니다."
                display_text = f"'{word}' 헤더의 특정 값이 {predicted_label_context_for_meaning} {('증가시키는' if weight > 0 else '감소시키는')} 영향을 미쳤습니다."
            
            # --- 일반적인 URL 구성 요소 (도메인, 경로 조각, 쿼리 파라미터 등) ---
            # 숫자로만 이루어진 단어는 너무 흔하고 의미 없으므로 제외 
            elif ('.' in word and len(word) > 2) or \
                 (len(word) <= 30 and all(c.isalnum() or c in ['-', '_', '%', '/', '.'] for c in word) and not word.isdigit()):
                meaning = "URL의 경로나 쿼리 파라미터에 포함된 문자열 패턴은 악성 행위(예: 피싱, 멀웨어 배포)를 숨기거나 유도하기 위해 비정상적으로 구성되는 경우가 많습니다. 비정상적인 길이, 반복되는 문자열, 인코딩된 문자열 등이 여기에 해당할 수 있습니다."
                display_text = f"URL 내의 '{word}' 패턴이 {predicted_label_context_for_meaning} {('증가시키는' if weight > 0 else '감소시키는')} 영향을 미쳤습니다."
            
            return display_text, meaning

        # --- 요약 설명 생성 (reason_summary) ---
        significant_features_for_summary = []
        for word, weight in explanation_list:
            if abs(weight) > 0.05: # 좀 더 높은 임계값을 사용하여 주요 특징만 요약에 포함
                # 요약에는 키워드 자체만 명시적으로 포함 (LIME이 분석한 단어 그대로)
                # desc_text가 None이 아닌 경우에만 추가
                desc_text, _ = get_understandable_explanation_text_and_meaning(word, weight, "") # 실제 설명 생성은 아니고, 필터링 목적으로 호출
                if desc_text: # 필터링된 단어 중 유효한 것만 추가
                    
                    significant_features_for_summary.append(word)

        # 요약 문구에 HTTP 헤더 정보 부재가 중요한 악성 신호일 경우 포함
        if predicted_label == 'malicious':
            if 'NOHEADER' in [f[0].lower() for f in explanation_list if abs(f[1]) > 0.05]: # NOHEADER가 주요 특징으로 꼽혔을 때
                reason_phrases.append(f"특히 **HTTP 헤더 정보 부재**와 '{', '.join(significant_features_for_summary[:2])}' 등과 같이 **정상적이지 않은 패턴**이 악성으로 의심됩니다.")
            elif significant_features_for_summary:
                reason_phrases.append(f"특히 '{', '.join(significant_features_for_summary[:3])}' 등과 같이 **정상적이지 않은 패턴**이 악성으로 의심됩니다.")
            else:
                reason_phrases.append("URL의 전반적인 구조와 패턴이 알려진 악성 URL과 유사하여 의심됩니다.")
        else: # 'benign'
            if significant_features_for_summary:
                reason_phrases.append(f"이 URL은 '{', '.join(significant_features_for_summary[:3])}' 등과 같이 **일반적이고 신뢰할 수 있는 패턴**을 가지고 있어 안전합니다.")
            else:
                reason_phrases.append("URL의 전반적인 구조와 패턴이 알려진 정상 URL과 유사하여 안전합니다.")

        reason_summary = "이 URL은 " + predicted_label.upper() + "로 판단됩니다. " + " ".join(reason_phrases)

        # --- 상세 설명 출력 ---
        print("\n--- 상세 분석 (URL 특징별 기여도) ---")
        print("💡 이 섹션에서는 모델이 URL을 분석하며 중요하게 판단한 주요 특징들과 그 이유를 설명합니다.")
        print("   '영향도'는 각 특징이 최종 판단에 얼마나 강하게 기여했는지를 숫자로 나타냅니다. (값이 클수록 기여도 높음)\n")
        
        has_understandable_explanation = False
        for word, weight in explanation.as_list(label=predicted_class_id): 
            classification_impact_for_meaning = ""
            if predicted_label == 'malicious':
                classification_impact_for_meaning = "악성으로 판단될 가능성을" if weight > 0 else "정상으로 판단될 가능성을"
            else: # benign
                classification_impact_for_meaning = "정상으로 판단될 가능성을" if weight > 0 else "악성으로 판단될 가능성을"

            desc_text, meaning = get_understandable_explanation_text_and_meaning(word, weight, classification_impact_for_meaning)
            
            if desc_text and abs(weight) > 0.01: # 영향도가 0.01 이상인 경우만 출력
                print(f"  - **특징**: {desc_text} (영향도: {weight:.4f})")
                if meaning:
                    print(f"    **설명**: {meaning}\n")
                has_understandable_explanation = True
        
        if not has_understandable_explanation:
            print("자동 필터링된 주요 특징은 없습니다. 원본 LIME 결과에는 복잡한 문자열이나 미미한 영향이 포함될 수 있습니다.")
        print("----------------------------")
        print(" 이 분석은 모델이 학습한 내용을 바탕으로 하며, 모든 URL에 대한 절대적인 판단 기준은 아닙니다. ")
        print("   의심스러운 URL은 직접 접속하기 전 반드시 주의하시기 바랍니다.\n")


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
        "detailed_explanation": explanation_list # 원본 LIME 결과를 반환
    }