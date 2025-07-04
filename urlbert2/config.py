# /home/kong/urlbert/url_bert/urlbert2/config.py
import os
import torch

# 프로젝트 루트 디렉토리 설정: 이 파일(config.py)의 상위 디렉토리가 프로젝트 루트입니다.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- 1. 경로 설정 ---
# os.path.join을 사용하여 OS에 독립적인 경로를 생성합니다.
# PROJECT_ROOT를 기준으로 모든 경로를 설정하여 상대 경로 문제를 해결합니다.

BERT_TOKENIZER_DIR = os.path.join(PROJECT_ROOT, 'bert_tokenizer')
VOCAB_FILE_PATH = os.path.join(BERT_TOKENIZER_DIR, 'vocab.txt')

BERT_CONFIG_DIR = os.path.join(PROJECT_ROOT, 'bert_config')

BERT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'bert_model')
BERT_PRETRAINED_MODEL_PATH = os.path.join(BERT_MODEL_DIR, 'urlBERT (1).pt')

CLASSIFIER_CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'finetune', 'phishing', 'checkpoints')
CLASSIFIER_MODEL_PATH = os.path.join(CLASSIFIER_CHECKPOINTS_DIR, 'modelx_URLBERT_80.pth')

# --- 2. 모델 및 학습 관련 설정 ---
SEED = 42 # <-- 이 부분이 중요합니다! config에 SEED가 정의되어야 합니다.
PAD_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT 모델 설정 시 필요한 kwargs (원본 url.py의 config_kwargs와 동일)
BERT_CONFIG_KWARTS = {
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
    "hidden_dropout_prob": 0.1,
    "vocab_size": 5000, # 모델 학습 시 사용했던 vocab_size와 동일해야 합니다.
}

# 분류 클래스 라벨 (모델 출력 인덱스와 매칭)
CLASS_LABELS = {0: 'benign', 1: 'malicious'}

# HTTP 헤더 추출 시 중요한 헤더 목록 (원본 url.py와 동일)
IMPORTANT_HEADERS = ["Server", "Content-Type", "Set-Cookie", "Location", "Date"]

# --- 3. 기타 설정 (LIME 관련 설정도 여기에 포함) ---
REQUEST_TIMEOUT_SECONDS = 5 # requests.get의 timeout
LIME_NUM_FEATURES = 5
LIME_NUM_SAMPLES = 1000