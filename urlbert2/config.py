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
PAD_SIZE=512
SEED = 42 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT 모델 설정 시 필요한 kwargs 
BERT_CONFIG_KWARTS = {
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
    "hidden_dropout_prob": 0.1,
    "vocab_size": 5000, 
}

# 분류 클래스 라벨 
CLASS_LABELS = {0: 'benign', 1: 'malicious'}

# HTTP 헤더 추출 시 중요한 헤더 목록 
IMPORTANT_HEADERS = ["Server", "Content-Type", "Set-Cookie", "Location", "Date"]

# --- 3. 기타 설정 ---
REQUEST_TIMEOUT_SECONDS = 5 # requests.get의 timeout
LIME_NUM_FEATURES = 5
LIME_NUM_SAMPLES = 1000


TRUSTED_DOMAINS_FOR_EXPLANATION = [
    'google', 'naver', 'kakao', 'daum', 'youtube', 'facebook', 'instagram', 
    'apple', 'microsoft', 'amazon', 'paypal', 'coupang', 'gmarket', '11st',
    'tistory', 'wordpress', 'github', 'gitlab', 'stackoverflow', 'twitter', 'x.com',
    'wikipedia', 'netflix', 'ebay', 'alibaba', 'shopify', 'bbc', 'cnn', 'discord',
    'reddit', 'linkedin', 'line', 'telegram', 'vimeo', 'unsplash', 'pixabay', 'flickr',
    'domain_name','facebook.com','google.com','akadns.net','gstatic.com','microsoft.com',
    'apple.com','microsoftonline.com','bing.com','googleusercontent.com','whatsapp.net',
    'doubleclick.net','aaplimg.com','aa-rt.sharepoint.com','office.com','digicert.com',
    'netgate.net.nz','play.googleapis.com','googleapis.com','www.googleapis.com','fbcdn.net','akamai.com',
    'update.googleapis.com','akamai-zt.com','msn.com','akamai.net','skype.com','office.net','icloud.com',
    'safebrowsing.googleapis.com','windows.com','tiktokcdn.com','officeapps.live.com','app-measurement.com',
    'windowsupdate.com','instagram.com','apple-dns.net','youtube.com','ytimg.com','clientservices.googleapis.com',
    'akamaiedge.net','windows.net','googleadservices.com','googlesyndication.com','amazonaws.com',
    'google-analytics.com','ntp.org','login.live.com','cloudflare.com','googlevideo.com',
    'office365.com','sparkbb.co.nz','optimizationguide-pa.googleapis.com','gvt1.com','akamaized.net',
    'azureedge.net','fonts.googleapis.com','adobe.com','googletagmanager.com','content-autofill.googleapis.com',
    'gvt2.com','lencr.org','azure.com','youtubei.googleapis.com','play-fe.googleapis.com',
    'tiktokv.com','msd.govt.nz','sfx.ms','churchofjesuschrist.org','maerskgroup.com','android.googleapis.com',
    'outlook.com','global-gateway.net.nz','notifications-pa.googleapis.com','crashlytics.com',
    'pki.goog','appsflyersdk.com','ggpht.com','cloudsink.net','tiktokrow-cdn.com','whatsapp.com','yahoo.com',
    'static.microsoft','fastly-edge.com','android.com','cdn-apple.com','adobedtm.com','firebaseremoteconfig.googleapis.com',
    'firebaselogging-pa.googleapis.com','scorecardresearch.com','prod-lt-playstoregatewayadapter-pa.googleapis.com',
    'netflix.com','cisco.com','facebook.net','dell.com','adnxs.com','amazon-adsystem.com','demdex.net',
    'playstoregatewayadapter-pa.googleapis.com','zscalertwo.net','sentry.io'
    ]
 