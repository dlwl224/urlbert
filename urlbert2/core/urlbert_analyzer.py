

import os
import sys
import torch
import torch.nn.functional as F
import requests
import random
import numpy as np
import re
from urllib.parse import urlparse # URL 파싱을 위해 추가

from pytorch_pretrained_bert import BertTokenizer
from lime.lime_text import LimeTextExplainer


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from config import (
    PAD_SIZE, DEVICE, CLASS_LABELS, IMPORTANT_HEADERS,
    REQUEST_TIMEOUT_SECONDS, LIME_NUM_FEATURES, LIME_NUM_SAMPLES,
    TRUSTED_DOMAINS_FOR_EXPLANATION # 신뢰할 수 있는 도메인 목록
)

# 현재 파일의 디렉토리 (core)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉토리 (urlbert2)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 프로젝트 루트 디렉토리를 Python 검색 경로에 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
        
        important = {
            k: resp_headers.get(k, "") for k in IMPORTANT_HEADERS
        }
        header_str = ", ".join(f"{k}: {v}" for k, v in important.items() if v)
        return header_str if header_str else "NOHEADER"
    except requests.exceptions.RequestException:
        return "NOHEADER"
    except Exception:
        return "NOHEADER"

# --- 데이터 전처리 함수 ---
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


# --- 1. 모델 예측만 수행하는 함수 ---
def predict_url(url: str, model, tokenizer) -> dict:
    header_info = get_header_info(url)
    
    input_ids, input_types, input_masks = preprocess_url_for_inference(
        url, header_info, tokenizer, PAD_SIZE
    )

    with torch.no_grad():
        outputs = model([input_ids, input_types, input_masks])
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_id = torch.argmax(probabilities, dim=1).item()

    predicted_label = CLASS_LABELS[predicted_class_id]
    confidence = probabilities[0][predicted_class_id].item() # 0~1 사이 값으로 반환
    
    return {
        "predicted_label": predicted_label,
        "confidence": confidence, # 0~1 사이 값으로 반환
        "predicted_class_id": predicted_class_id,
        "header_info": header_info # LIME 설명을 위해 헤더 정보도 함께 반환
    }

# --- 2. LIME 설명을 생성하는 함수 ---
def explain_prediction_with_lime(url: str, header_info: str, model, tokenizer, predicted_class_id: int) -> dict:
    predicted_label = CLASS_LABELS[predicted_class_id]
    full_text_for_lime = f"{url} [SEP] {header_info}"
    
    explanation_list = []
    reason_summary = f"판단 근거를 설명하는 중 오류가 발생했습니다."

    # URL을 한 번만 파싱하여 함수 전체에서 사용
    parsed_url = urlparse(url)
    netloc_lower = parsed_url.netloc.lower()
    tld = parsed_url.netloc.split('.')[-1] if '.' in parsed_url.netloc else ''

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
        
        # --- LIME 설명 요약 및 상세 설명 로직 개선 ---
        def get_understandable_explanation_text_and_meaning(word, weight, current_predicted_label, original_url, parsed_url_info, netloc_lower_info, tld_info):
            word_lower = word.strip().lower() # 여기서 word_lower 정의 및 사용
            
            # 1단계 필터링: BERT 특수 토큰 및 LIME 노이즈 필터링 강화
            # 단일 특수문자, 짧은 숫자, 흔한 세션 쿠키 등은 LIME이 의미 없게 뽑아낼 수 있으므로 필터링
            if word_lower in ["[sep]", "[cls]", "[pad]", "sep", "cls", "pad", "##s", "com", "co", "kr", "net", "org", "io", "ai", "app", "ly", "me", "biz", "info", "name", 
                              "php", "html", "asp", "aspx", "htm", "default", "index", # 흔한 파일명/확장자
                              "session", "cookie", "id", "data", # 일반적인 파라미터 이름
                              "www"] or \
               len(word_lower) > 50 or \
               re.fullmatch(r'^[!@#$%^&*()_+=\[\]{}|;:\'",.<>?`~]$', word_lower) or \
               (word_lower.isdigit() and len(word_lower) < 4 and not re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', word_lower)) or \
               (word_lower in TRUSTED_DOMAINS_FOR_EXPLANATION and current_predicted_label == 'malicious' and weight > 0) : # 신뢰 도메인이 악성으로 판단되는데 강하게 기여하는 경우 필터링 (오류 방지)
                return None, None
            
            display_text = ""
            meaning = ""
            
            # 인자로 받은 URL 파싱 정보 사용
            netloc_lower_local = netloc_lower_info
            tld_local = tld_info
            
            # 신뢰할 수 있는 TLD 목록 (TRUSTED_DOMAINS_FOR_EXPLANATION에서 추출)
            trusted_tlds_from_config = [td.split('.')[-1] for td in TRUSTED_DOMAINS_FOR_EXPLANATION if '.' in td]
            
            
            
            if word_lower == 'noheader':
                display_text = f"**HTTP 헤더 정보 부재**"
                if current_predicted_label == 'malicious' and weight > 0: # 악성인데 기여도 양수: 헤더 부재가 악성 판단에 긍정적 기여
                    meaning = "URL 접속 시 **HTTP 헤더 정보가 전혀 없거나 매우 불완전한 경우**, 이는 서버가 정보를 숨겨 분석을 어렵게 하거나 비정상적인 동작을 시도할 수 있음을 나타내 **악성으로 판단하는 중요한 근거**가 됩니다. 정상 웹사이트는 다양한 헤더 정보를 제공합니다."
                elif current_predicted_label == 'benign' and weight < 0: # 정상인데 기여도 음수: 헤더 부재가 정상 판단에 부정적 영향
                    meaning = "이 URL은 정상으로 판단되었지만, **HTTP 헤더 정보가 없거나 매우 불완전하여** 정상 판단에 다소 부정적인 영향을 주었습니다. 정상적인 웹사이트는 보통 다양한 헤더 정보를 주고받습니다."
                else: # 그 외의 경우 (예: 악성인데 음수 기여, 정상인데 양수 기여 등)
                    meaning = "이 URL에 대한 HTTP 헤더 정보가 없다는 사실이 모델의 판단에 영향을 미쳤습니다. 헤더 정보 부재는 때때로 의심스러운 활동과 관련될 수 있습니다."
                return display_text, meaning
        
            # --- 긍정적 기여 요인 (정상 URL 판단에 중요) ---
            if current_predicted_label == 'benign' and weight > 0:
                # HTTPS 프로토콜
                if word_lower == 'https':
                    display_text = f"안전한 **HTTPS 연결**"
                    meaning = "HTTPS는 웹사이트와 사용자 간의 통신을 암호화하여 데이터를 안전하게 보호하는 프로토콜입니다. 이 URL이 HTTPS를 사용한다는 점은 **보안성을 높이는 긍정적인 신호**로 작용했습니다."
                # 신뢰할 수 있는 도메인 또는 TLD
                elif any(td_part in word_lower for td_part in TRUSTED_DOMAINS_FOR_EXPLANATION) or \
                     any(td_part in netloc_lower_local for td_part in TRUSTED_DOMAINS_FOR_EXPLANATION) or \
                     (tld_local and (tld_local in trusted_tlds_from_config)):
                    display_text = f"'{word}'와 같은 **신뢰할 수 있는 도메인 또는 TLD**"
                    meaning = "이 URL이 **널리 알려진 신뢰할 수 있는 도메인 또는 TLD**를 포함하고 있어 정상으로 판단될 가능성을 높였습니다."
                
                elif abs(weight) > 0.05 and len(word_lower) > 1 and not word_lower.isdigit() and not re.fullmatch(r'.*\d{5,}.*', word_lower): # 너무 긴 숫자열 제외
                    display_text = f"URL 내 '{word}' 패턴"
                    meaning = f"URL 경로 또는 쿼리 파라미터에 포함된 '{word}'와 같은 **일반적이고 예상 가능한 패턴**이 URL의 정상성을 나타내는 긍정적인 신호로 작용했습니다."
            
            # --- 부정적 기여 요인 (악성 URL 판단에 중요) ---
            elif current_predicted_label == 'malicious' and weight > 0:
                # HTTP 프로토콜 (HTTPS가 아닌 경우)
                if word_lower == 'http': 
                    display_text = f"보안에 취약한 **HTTP 프로토콜**"
                    meaning = "HTTP는 암호화되지 않은 통신 프로토콜로 데이터가 노출될 위험이 있습니다. 이 URL이 HTTP를 사용한다는 점은 **보안 취약성을 시사하며 악성으로 판단하는 주된 근거** 중 하나입니다."
                # IP 주소 (도메인 대신 IP 직접 사용)
                elif re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', word_lower):
                    display_text = f"URL 내 **IP 주소 직접 사용**"
                    meaning = "일반적인 웹사이트는 도메인 이름을 사용하지만, 이 URL은 **IP 주소를 직접 사용하여 접속을 유도**하고 있습니다. 이는 추적 회피나 의심스러운 목적을 숨기려는 악성 URL의 흔한 특징입니다."
                # 의심스러운 최상위 도메인 (TLD)
                elif tld_local and (tld_local in ['vip', 'xin', 'top', 'xyz', 'online', 'loan', 'click', 'site', 'bid', 'asia', 'it'] or \
                                   (2 <= len(tld_local) <= 5 and not tld_local.isalnum() and not tld_local.isdigit() and tld_local not in trusted_tlds_from_config)): 
                    display_text = f"'.{tld_local}'와 같은 **의심스러운 최상위 도메인**"
                    meaning = f"URL의 '.{tld_local}'와 같은 최상위 도메인(TLD)은 **스팸, 피싱, 악성코드 유포 등에 자주 사용**되는 경향이 있어, 이 URL을 악성으로 판단하는 **강력한 근거**가 됩니다."
                # 비정상적이거나 난독화된 서브도메인/경로 패턴 (예: "8noX2wTHr", "com-xaaawn")
                elif (re.search(r'-\w{5,}|[a-f0-9]{8,}', word_lower) and len(word_lower) > 10) or \
                     (len(word_lower) > 15 and (word_lower.isdigit() or re.fullmatch(r'^[a-zA-Z0-9]+$', word_lower))) or \
                     re.search(r'\d{3,}\.\d{3,}\.\d{3,}', word_lower) or \
                     ('//' in word_lower and word_lower.count('/') > 3) or \
                     (re.search(r'([a-zA-Z0-9]{2,}\.){2,}[a-zA-Z]{2,}', word_lower) and not any(td in word_lower for td in TRUSTED_DOMAINS_FOR_EXPLANATION)): 
                    display_text = f"URL 내 '{word}'와 같은 **비정상적이거나 난독화된 문자열/서브도메인**"
                    meaning = "이 URL은 주소에 **의미 없는 긴 문자열, 무작위 숫자/문자 조합, 비정상적인 서브도메인 구조** 등을 사용하여 의심스러운 의도를 숨기려 하고 있으며, 이는 악성 URL의 **전형적인 특징**입니다."
                # 중요한 HTTP 헤더 이름 및 값 (악성으로 기여) - 필요시 추가
                elif any(header_name.lower() in word_lower for header_name in IMPORTANT_HEADERS if word_lower != 'noheader'):
                    display_text = f"'{word}'와 같은 **비정상적인 HTTP 헤더 패턴**"
                    meaning = f"HTTP 헤더 '{word}'의 존재 또는 값이 일반적이지 않거나, **악성 서버에서 흔히 발견되는 패턴**과 일치하여 URL의 악성도를 높이는 요인으로 작용했습니다."
                # 피싱 관련 키워드 (login, admin, bank, secure 등)
                elif word_lower in ['login', 'admin', 'bank', 'secure', 'update', 'verify', 'account'] and abs(weight) > 0.05:
                    display_text = f"URL 내 '{word}'와 같은 **피싱 의심 키워드**"
                    meaning = f"이 URL은 '{word}'와 같이 **금융 기관, 로그인 페이지 등을 위장하려는 피싱 공격에 자주 사용되는 키워드**를 포함하고 있어 악성으로 판단하는 중요한 근거가 됩니다."
                # 다운로드 가능한 파일 확장자
                elif any(word_lower.endswith(ext) for ext in ['.exe', '.zip', '.rar', '.doc', '.docx', '.xls', '.xlsx', '.pdf', '.js']) and abs(weight) > 0.05:
                    display_text = f"URL 내 '{word}'와 같은 **악성 파일 확장자**"
                    meaning = f"이 URL은 실행 파일이나 압축 파일, 문서 파일 등 **악성코드 유포에 흔히 사용되는 파일 확장자**를 포함하고 있어 악성으로 판단되는 요인입니다."
                # 단축 URL (short.ly/malware 와 같이 단축 도메인 자체)
                elif word_lower in ['short.ly', 'bit.ly', 'tinyurl.com', 'goo.gl', 'buff.ly', 'ow.ly', 't.co'] and abs(weight) > 0.05:
                    display_text = f"'{word}'와 같은 **단축 URL 도메인**"
                    meaning = "단축 URL은 실제 목적지 주소를 숨겨 악성 링크를 유포하는 데 자주 사용됩니다. 이 URL이 단축 URL 서비스를 이용했다는 점이 악성 판단에 기여했습니다."
                # 기타 악성으로 판단하는 일반 패턴 (기여도가 높고 위 항목에 해당하지 않는 경우)
                elif abs(weight) > 0.05 and len(word_lower) > 1: # 너무 짧은 단어 제외
                    display_text = f"URL 내 '{word}'와 같은 **의심스러운 패턴**"
                    meaning = "이 URL 내에 모델이 악성 URL에서 자주 발견한 것으로 학습된 **특정 의심스러운 문자열 패턴**이 포함되어 있어 악성 판단에 기여했습니다."

            # --- 정상으로 판단되는데 방해하는 요인 (음수 기여도) ---
            elif current_predicted_label == 'benign' and weight < 0:
                # 보안에 취약한 HTTP 프로토콜 (정상으로 판단되는데 음수 기여)
                if word_lower == 'http':
                    display_text = f"보안에 취약한 **HTTP 프로토콜**"
                    meaning = "이 URL은 정상으로 판단되었지만, HTTP를 사용하여 통신이 암호화되지 않는다는 점은 **보안 취약 요소로 인식되어 정상 판단을 다소 방해**했습니다. 최신 웹사이트는 대부분 HTTPS를 사용합니다."
                
                # 의심스러운 패턴이 존재하지만 정상으로 분류된 경우 (오탐 가능성)
                elif abs(weight) > 0.05 and len(word_lower) >= 5 and (word_lower.isdigit() or re.fullmatch(r'^[a-zA-Z0-9]+$', word_lower) or re.fullmatch(r'.*\d{5,}.*', word_lower)):
                    display_text = f"URL 내 '{word}'와 같은 **의심스러운 패턴**"
                    meaning = "이 URL은 정상으로 판단되었지만, URL 내에 **악성 URL에서 흔히 발견되는 의심스러운 문자열 패턴**이 포함되어 있어 정상 판단을 다소 방해하는 요소로 작용했습니다."
                # 쿠키/세션 이름이 음수 기여를 하는 경우
                elif word_lower in ["aec", "nid", "sid", "phpsessid", "jsessionid", "ga"] and abs(weight) > 0.01:
                    display_text = f"'{word}'와 같은 **일반적인 쿠키/세션 패턴**"
                    meaning = "이 URL은 정상으로 판단되었지만, 특정 쿠키/세션 이름이 **악성 URL에서도 발견되는 경우가 있어** 모델의 정상 판단에 아주 미미하게 부정적인 영향을 미쳤을 수 있습니다."

            # --- 악성으로 판단되는데 완화하는 요인 (음수 기여도) ---
            elif current_predicted_label == 'malicious' and weight < 0:
                # 신뢰할 수 있는 도메인 (악성인데 음수 기여)
                if any(td_part in word_lower for td_part in TRUSTED_DOMAINS_FOR_EXPLANATION) or \
                   any(td_part in netloc_lower_local for td_part in TRUSTED_DOMAINS_FOR_EXPLANATION): 
                    display_text = f"'{word}'와 같은 **신뢰할 수 있는 도메인**"
                    meaning = "이 URL은 악성으로 판단되었지만, **신뢰할 수 있는 도메인**이 포함되어 있어 악성 판단을 다소 완화하는 요인으로 작용했습니다. 이는 악성 URL이 정상 사이트를 모방하거나 리다이렉션을 활용하는 경우에 나타날 수 있습니다."
                # HTTPS 프로토콜 (악성인데 음수 기여)
                elif word_lower == 'https':
                    display_text = f"안전한 **HTTPS 연결**"
                    meaning = "이 URL은 악성으로 판단되었지만, HTTPS를 사용하여 통신을 암호화하고 있습니다. 이는 악성 URL이 **합법적인 것처럼 보이게 하려는 시도**일 수 있으며, 악성 판단을 다소 완화하는 요인으로 작용했습니다."
                # WWW 접두사 (악성인데 음수 기여, 기여도 높을 때만)
                elif word_lower == 'www' and abs(weight) > 0.05:
                    display_text = f"'WWW' 접두사"
                    meaning = "이 URL은 악성으로 판단되었지만, 'WWW' 접두사가 포함되어 있어 악성 판단을 다소 완화하는 요인으로 작용했습니다."
                # 기타 정상으로 판단되는 데 기여할 수 있는 일반 패턴 (악성인데 음수 기여)
                elif abs(weight) > 0.05 and len(word_lower) > 1 and not word_lower.isdigit() and not re.fullmatch(r'.*\d{5,}.*', word_lower):
                    display_text = f"URL 내 '{word}'와 같은 **정상적인 패턴**"
                    meaning = "이 URL은 악성으로 판단되었지만, URL 내에 **일반적으로 정상 URL에서 발견되는 패턴**이 포함되어 있어 악성 판단을 다소 완화하는 요소로 작용했습니다."

            return display_text, meaning

        # --- 상세 설명 출력 포매팅 ---
        detailed_explanation_output = []
        detailed_explanation_output.append("\n--- 상세 분석 (URL 특징별 기여도) ---")
        detailed_explanation_output.append("💡 모델이 URL을 분석하며 중요하게 판단한 주요 특징들과 그 이유를 설명합니다.")
        detailed_explanation_output.append("   '기여도'는 각 특징이 최종 판단에 얼마나 강하게 기여했는지를 숫자로 나타냅니다. (값이 클수록 기여도 높음, 음수일수록 예측에 반대되는 영향)\n")
        
        has_understandable_explanation = False
        significant_features_for_summary = [] # 요약에 사용될 주요 특징 목록

        # 정렬: 절대 기여도가 높은 순서대로
        sorted_explanation = sorted(explanation_list, key=lambda x: abs(x[1]), reverse=True)

        for word, weight in sorted_explanation:
            word_lower = word.lower() # 여기서 word_lower를 정의하고 사용합니다.

            # LIME이 뽑아낸 토큰이 원래 URL에 포함되어 있는지 또는 헤더에서 온 'NOHEADER'인지 확인
            is_from_url = word_lower in url.lower() or \
                          (parsed_url.netloc and word_lower in parsed_url.netloc.lower()) or \
                          (parsed_url.path and word_lower in parsed_url.path.lower()) or \
                          (parsed_url.query and word_lower in parsed_url.query.lower())
            
            is_noheader_word = (word_lower == 'noheader') # 'noheader'인 경우 별도 플래그

            # get_understandable_explanation_text_and_meaning 함수 호출 시 파싱된 정보를 인자로 전달
            desc_text, meaning = get_understandable_explanation_text_and_meaning(word, weight, predicted_label, url, parsed_url, netloc_lower, tld)
            
            # 유의미한 기여도 (절대값 0.01 이상)와 설명이 있는 경우에만 포함
            if desc_text and abs(weight) >= 0.01: # 낮은 기여도 필터링 유지
                # 'NOHEADER'는 URL에서 온 것은 아니므로 is_from_url을 무시하고 항상 포함
                if is_noheader_word:
                    detailed_explanation_output.append(f"  - **특징**: {desc_text} (기여도: {weight:.4f})")
                    if meaning:
                        detailed_explanation_output.append(f"    **설명**: {meaning}\n")
                    has_understandable_explanation = True
                    # 요약에 사용될 특징 추가 (최대 3개)
                    if len(significant_features_for_summary) < 3:
                        significant_features_for_summary.append("헤더 부재")
                    continue
                
                # URL에서 온 특징만 유의미하게 설명 (헤더 키/값 제외)
                # IMPORTANT_HEADERS는 헤더 정보에서 오는 것이므로 is_from_url 조건에서 제외
                elif (is_from_url or any(h.lower() in word_lower for h in IMPORTANT_HEADERS)) and \
                     not any(h.lower() == word_lower for h in IMPORTANT_HEADERS) and \
                     not (word_lower in TRUSTED_DOMAINS_FOR_EXPLANATION and predicted_label == 'malicious' and weight > 0) : # 신뢰 도메인이 악성으로 판단되는데 강하게 기여하는 경우 제외
                    
                    detailed_explanation_output.append(f"  - **특징**: {desc_text} (기여도: {weight:.4f})")
                    if meaning:
                        detailed_explanation_output.append(f"    **설명**: {meaning}\n")
                    has_understandable_explanation = True

                    # 요약에 사용될 특징 추가 (최대 3개)
                    if len(significant_features_for_summary) < 3:
                        if "HTTPS 연결" in desc_text:
                            significant_features_for_summary.append("https")
                        elif "HTTP 프로토콜" in desc_text:
                            significant_features_for_summary.append("http")
                        elif "신뢰할 수 있는 도메인" in desc_text:
                            # 'google.com' 또는 'google'만 남기도록
                            if '.' in word: # word는 원본 단어 (소문자X)
                                try: # URL 파싱 시도 (word가 유효한 도메인 형태일 경우)
                                    parsed_sum_domain = urlparse("https://" + word).netloc
                                    significant_features_for_summary.append(parsed_sum_domain)
                                except ValueError: # 파싱 실패 시 원본 word 사용
                                    significant_features_for_summary.append(word)
                            else:
                                significant_features_for_summary.append(word) # word는 원본 단어 (소문자 X)
                        elif "'WWW' 접두사" in desc_text:
                            significant_features_for_summary.append("www")
                        elif "IP 주소 직접 사용" in desc_text:
                            significant_features_for_summary.append("IP 주소 사용")
                        elif "비정상적이거나 난독화된 문자열" in desc_text or "의심스러운 패턴" in desc_text or "의미 없는 긴 문자열" in desc_text:
                            if len(word) > 15:
                                significant_features_for_summary.append(word[:12] + "...") 
                            else:
                                significant_features_for_summary.append(word)
                        elif "쿠키/세션 패턴" in desc_text: # 이 항목은 설명을 위해 유지하되 요약에는 포함하지 않는 것을 고려
                            # significant_features_for_summary.append(f"{word} (쿠키 패턴)")
                            pass
                        elif "의심스러운 최상위 도메인" in desc_text or "특이한 최상위 도메인" in desc_text:
                            tld_extracted_from_desc = re.search(r"'\.([^']+)'", desc_text)
                            if tld_extracted_from_desc:
                                significant_features_for_summary.append(f".{tld_extracted_from_desc.group(1)}")
                            else:
                                significant_features_for_summary.append(word)
                        elif "피싱 의심 키워드" in desc_text:
                            significant_features_for_summary.append(word)
                        elif "악성 파일 확장자" in desc_text:
                            significant_features_for_summary.append(word)
                        elif "단축 URL 도메인" in desc_text:
                            significant_features_for_summary.append(word)
                        else: # 위 분류에 속하지 않는 일반적인 유의미한 단어
                            significant_features_for_summary.append(word)
                
        if not has_understandable_explanation:
            detailed_explanation_output.append("   모델이 판단에 사용한 주요 특징이 명확하게 필터링되지 않거나, 기여도가 낮은 특징들이 대부분입니다.")
            detailed_explanation_output.append("   이는 URL의 특징이 기존 학습 데이터와 유사하지 않거나, 미묘한 패턴들로 이루어진 경우 발생할 수 있습니다.\n")

        detailed_explanation_output.append("----------------------------")
        detailed_explanation_output.append("💡 이 분석은 모델이 학습한 내용을 바탕으로 하며, 모든 URL에 대한 절대적인 판단 기준은 아닙니다. ")
        detailed_explanation_output.append("   의심스러운 URL은 직접 접속하기 전 반드시 주의하시기 바랍니다.\n")

        # --- 요약 설명 생성 (reason_summary) ---
        reason_phrases = []
        if predicted_label == 'malicious':
            if significant_features_for_summary:
                malicious_patterns_display = []
                mitigating_patterns_display = []

                for feat in set(significant_features_for_summary): # 요약 특징 중복 제거
                    corresponding_weight = None
                    for lime_word, lime_weight in explanation_list:
                        # 요약용 feat과 LIME의 word를 매칭하는 더 견고한 로직
                        # word_lower를 사용하여 비교 일관성 확보
                        lime_word_lower = lime_word.lower()
                        if feat.lower() == lime_word_lower or \
                           (feat.endswith('...') and lime_word_lower.startswith(feat[:-3].lower())) or \
                           (feat.startswith('.') and f".{lime_word_lower}" == feat) or \
                           (feat == "헤더 부재" and lime_word_lower == 'noheader') or \
                           (feat == "IP 주소 사용" and re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', lime_word_lower)) or \
                           (feat == "https" and lime_word_lower == 'https') or \
                           (feat == "http" and lime_word_lower == 'http') or \
                           (feat == "www" and lime_word_lower == 'www') :
                            corresponding_weight = lime_weight
                            break
                    
                    if corresponding_weight is not None:
                        if corresponding_weight > 0: # 악성으로 강하게 기여
                            malicious_patterns_display.append(feat)
                        else: # 악성인데 완화 기여 (음수 기여도)
                            mitigating_patterns_display.append(feat)
                    # else: # LIME explanation_list에서 기여도를 찾지 못하면, 일단 악성 패턴으로 간주 (방어적 코드 - 위 매칭 로직 개선으로 이 else는 거의 실행 안 됨)
                    #     malicious_patterns_display.append(feat) 

                if malicious_patterns_display:
                    reason_phrases.append(f"특히 '{', '.join(sorted(list(set(malicious_patterns_display))))}' 등과 같이 **정상적이지 않거나 의심스러운 패턴**이 발견되어 악성으로 판단되었습니다.")
                
                if mitigating_patterns_display:
                    reason_phrases.append(f"이 URL은 '{', '.join(sorted(list(set(mitigating_patterns_display))))}'와 같은 일반적인 패턴을 포함하고 있으나, 전반적으로 악성 위험이 높은 것으로 분석되었습니다.")
                
                if not reason_phrases:
                    reason_phrases.append("URL의 전반적인 구조와 패턴이 알려진 악성 URL과 유사하여 의심됩니다.")

            else: # significant_features_for_summary가 비어있는 경우
                reason_phrases.append("URL의 전반적인 구조와 패턴이 알려진 악성 URL과 유사하여 의심됩니다.")
                
        else: # 'benign'
            if significant_features_for_summary:
                benign_patterns_display = []
                negative_patterns_display = []

                for feat in set(significant_features_for_summary): # 요약 특징 중복 제거
                    corresponding_weight = None
                    for lime_word, lime_weight in explanation_list:
                        # 요약용 feat과 LIME의 word를 매칭하는 더 견고한 로직
                        # word_lower를 사용하여 비교 일관성 확보
                        lime_word_lower = lime_word.lower()
                        if feat.lower() == lime_word_lower or \
                           (feat.endswith('...') and lime_word_lower.startswith(feat[:-3].lower())) or \
                           (feat.startswith('.') and f".{lime_word_lower}" == feat) or \
                           (feat == "헤더 부재" and lime_word_lower == 'noheader') or \
                           (feat == "IP 주소 사용" and re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', lime_word_lower)) or \
                           (feat == "https" and lime_word_lower == 'https') or \
                           (feat == "http" and lime_word_lower == 'http') or \
                           (feat == "www" and lime_word_lower == 'www') :
                            corresponding_weight = lime_weight
                            break

                    if corresponding_weight is not None:
                        if corresponding_weight > 0: # 정상으로 강하게 기여
                            benign_patterns_display.append(feat)
                        else: # 정상인데 방해 기여 (음수 기여도)
                            negative_patterns_display.append(feat)
                    # else: # LIME explanation_list에서 기여도를 찾지 못하면, 일단 정상 패턴으로 간주
                    #     benign_patterns_display.append(feat) 
                
                # 중복 제거 및 정렬
                benign_patterns_display = sorted(list(set(benign_patterns_display)))
                negative_patterns_display = sorted(list(set(negative_patterns_display)))


                if benign_patterns_display:
                    reason_phrases.append(f"이 URL은 '{', '.join(benign_patterns_display)}' 등과 같이 **일반적이고 신뢰할 수 있는 패턴**을 가지고 있어 안전합니다.")
                
                if negative_patterns_display:
                    reason_phrases.append(f"일부 '{', '.join(negative_patterns_display)}'와 같은 의심스러운 패턴이 발견되었으나, 전반적인 구조가 안전한 것으로 판단되었습니다.")
                
                if not reason_phrases:
                    reason_phrases.append("URL의 전반적인 구조와 패턴이 알려진 정상 URL과 유사하여 안전합니다.")
            else: # significant_features_for_summary가 비어있는 경우
                reason_phrases.append("URL의 전반적인 구조와 패턴이 알려진 정상 URL과 유사하여 안전합니다.")

        reason_summary = f"이 URL은 **{predicted_label.upper()}**로 판단됩니다. " + " ".join(reason_phrases)

    except Exception as e:
        print(f"LIME 설명 생성 중 오류 발생: {e}")
        reason_summary = f"이 URL은 **{predicted_label.upper()}**로 판단됩니다. 판단 근거를 설명하는 중 오류가 발생했습니다."
        explanation_list = []
        detailed_explanation_output = ["LIME 설명 생성 중 오류가 발생했습니다: " + str(e)]

    return {
        "reason_summary": reason_summary,
        "detailed_explanation_list": explanation_list, 
        "formatted_detailed_explanation": "\n".join(detailed_explanation_output)
    }

# --- 3. URL 분류 및 설명을 통합하는 함수 ---
def classify_url_and_explain(url: str, model, tokenizer) -> dict:
    # 1) URL 예측 수행
    pred_out = predict_url(url, model, tokenizer)

    # 2) LIME 설명 생성
    lime_out = explain_prediction_with_lime(
        url,
        pred_out["header_info"],
        model,
        tokenizer,
        pred_out["predicted_class_id"]
    )


    # 3) DB 저장용 필드명에 맞춰서 dict 반환
    is_mal = 1 if pred_out["predicted_label"] == "malicious" else 0

    return {
        "url": url,
        "header_info": pred_out["header_info"],
        "is_malicious": is_mal,
        "confidence": pred_out["confidence"],    # float 타입
        "true_label": None,                      
        "reason_summary": lime_out["reason_summary"],
        "detailed_explanation": lime_out["formatted_detailed_explanation"]
    }
