import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from transformers import AutoConfig, AutoModelForMaskedLM
import os

# --- 1. 설정 및 유틸리티 함수 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_SIZE = 512
VOCAB_FILE_PATH = "/home/kong/urlbert/url_bert/urlbert2/bert_tokenizer/vocab.txt"
BERT_CONFIG_PATH = "/home/kong/urlbert/url_bert/urlbert2/bert_config"
BERT_MODEL_PATH = "/home/kong/urlbert/url_bert/urlbert2/bert_model/urlBERT (1).pt"
CLASSIFIER_MODEL_PATH = '/home/kong/urlbert/url_bert/urlbert2/finetune/phishing/checkpoints/modelx_URLBERT_80.pth' # 학습된 모델 가중치 경로

IMPORTANT_HEADERS = ["Server", "Content-Type", "Set-Cookie", "Location", "Date"]

def get_header_info(url: str) -> str:
    """
    주어진 URL에서 중요한 HTTP 헤더 정보를 추출합니다.
    URL 접근 실패 시 "NOHEADER"를 반환합니다.

    :param url: 헤더 정보를 추출할 URL.
    :return: 쉼표로 구분된 중요한 헤더 정보 문자열 또는 "NOHEADER".
    """
    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/114 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/14.0.3 Safari/605.1.15",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 Version/14.0 Mobile/15E148 Safari/604.1"
        ]),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
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


def preprocess_url_for_inference(url: str, header_info: str, tokenizer: BertTokenizer, pad_size: int = 512):
    """
    URL과 추출된 헤더 정보를 BERT 모델 입력 형식으로 전처리합니다.

    :param url: 원본 URL.
    :param header_info: 추출된 HTTP 헤더 정보.
    :param tokenizer: BERT Tokenizer 인스턴스.
    :param pad_size: 패딩할 시퀀스 길이.
    :return: input_ids, input_types, input_masks 텐서.
    """
    text = f"{url} [SEP] {header_info}"
    tokenized_text = tokenizer.tokenize(text)
    
    # [CLS]와 [SEP] 토큰 추가
    tokens = ["[CLS]"] + tokenized_text + ["[SEP]"]

    # 토큰을 ID로 변환
    ids = tokenizer.convert_tokens_to_ids(tokens)
    types = [0] * (len(ids)) # 단일 시퀀스이므로 모두 0
    masks = [1] * len(ids)

    # 패딩 또는 잘라내기
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

# --- 2. 모델 클래스 정의 (기존 코드와 동일) ---
class BertForSequenceClassification(nn.Module):
    def __init__(self, bert, freeze=False):
        super(BertForSequenceClassification, self).__init__()
        self.bert = bert
        # 모든 BERT 파라미터는 학습 가능하게 설정
        for name, param in self.bert.named_parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 2) # BERT hidden size는 768, 2개 클래스 (benign, malicious)

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]
        outputs = self.bert(context, attention_mask=mask, token_type_ids=types, output_hidden_states=True)
        # [CLS] 토큰에 해당하는 마지막 레이어의 hidden state 사용
        hidden_states = outputs.hidden_states[-1][:,0,:]
        out = self.dropout(hidden_states)
        out = self.classifier(out)
        return out

# --- 3. 모델 로드 ---
print("모델 및 토크나이저 로드 중...")

# Tokenizer 로드
tokenizer = BertTokenizer(VOCAB_FILE_PATH)

# BERT Config 로드
config_kwargs = {
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
    "hidden_dropout_prob": 0.1,
    "vocab_size": 5000, # 학습 시 사용했던 vocab_size와 동일해야 함
}
config = AutoConfig.from_pretrained(BERT_CONFIG_PATH, **config_kwargs)

# AutoModelForMaskedLM 로드 (MaskedLM 헤드는 추론 시 사용하지 않으므로 제거)
bert_model_for_loading = AutoModelForMaskedLM.from_config(config=config)
bert_model_for_loading.resize_token_embeddings(config_kwargs["vocab_size"])

# 기존 BERT 모델의 가중치 로드
bert_dict = torch.load(BERT_MODEL_PATH, map_location=torch.device("cpu"))
bert_model_for_loading.load_state_dict(bert_dict)

# 분류 모델 초기화 및 학습된 가중치 로드
model = BertForSequenceClassification(bert_model_for_loading)
model.bert.cls = nn.Sequential() # MaskedLM 헤드 제거
model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval() # 추론 모드 설정

print("모델 로드 완료.")

# --- 4. 추론 함수 ---
def predict_url(url: str) -> str:
    """
    단일 URL을 입력받아 악성/정상 여부를 추론합니다.

    :param url: 분류할 URL.
    :return: 'malicious' 또는 'benign'.
    """
    print(f"\nURL 분석 시작: {url}")
    # 1. 헤더 추출
    header_info = get_header_info(url)
    print(f"추출된 헤더 정보: {header_info if header_info != 'EMPTY' else '없음'}")

    # 2. 데이터 전처리
    input_ids, input_types, input_masks = preprocess_url_for_inference(
        url, header_info, tokenizer, PAD_SIZE
    )

    # 3. 모델 추론
    with torch.no_grad():
        outputs = model([input_ids, input_types, input_masks])
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_id = torch.argmax(probabilities, dim=1).item()

    # 4. 결과 출력
    class_labels = {0: 'benign', 1: 'malicious'}
    predicted_label = class_labels[predicted_class_id]
    confidence = probabilities[0][predicted_class_id].item() * 100

    print(f"분류 결과: **{predicted_label.upper()}** (확신도: {confidence:.2f}%)")
    return predicted_label

# --- 사용 예시 ---
if __name__ == "__main__":
    # 정상이 예상되는 URL
    predict_url("https://www.google.com")
    predict_url("https://www.naver.com")

    # 악성이 예상되는 URL (예시 데이터에서 가져옴)
    predict_url("https://flowto.it/8noX2wTHr?fc=0")
    predict_url("https://ezdriverma.com-xaaawn.vip/")
    predict_url("https://com-pif.xin/")
    predict_url("https://www.c.cdnhwc6.com")

    # 존재하지 않거나 응답 없는 URL
    predict_url("http://thisurldoesnotexist12345.com")
    predict_url("http://192.168.1.99:8080/nonexistent") # 로컬 IP (접근 불가능)
    