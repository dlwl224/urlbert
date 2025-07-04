import pandas as pd
import requests
import time
import random
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random

IMPORTANT_HEADERS = ["Server", "Content-Type", "Set-Cookie", "Location", "Date"]

def get_header_info(url):
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
        response = requests.get(url, headers=headers, timeout=5)
        resp_headers = response.headers

        important = {
            k: resp_headers.get(k, "") for k in IMPORTANT_HEADERS
        }

        header_str = ", ".join(f"{k}: {v}" for k, v in important.items() if v)
        return header_str if header_str else "EMPTY"

    except:
        return "NOHEADER"


# 경로 설정
train_csv_path = "/mnt/c/Users/DS/Desktop/PythonProject/url_bert/url_bert/urlbert2/dataset/train.csv"
input_file = train_csv_path
output_dir = os.path.dirname(train_csv_path)
batch_dir = os.path.join(output_dir, "batches")
final_output_file = os.path.join(output_dir, "urlbert_input.csv")

# CSV 불러오기
df = pd.read_csv(input_file)
total = len(df)
batch_size = 10000
num_batches = (total + batch_size - 1) // batch_size

os.makedirs(batch_dir, exist_ok=True)

for batch_idx in range(num_batches):
    start = batch_idx * batch_size
    end = min(start + batch_size, total)
    batch_df = df.iloc[start:end].copy()
    
    print(f"Batch {batch_idx + 1}/{num_batches} 시작")

    urls = batch_df["url"].tolist()

    # 병렬 처리: 4~8개 프로세스 사용
    with Pool(processes=min(8, cpu_count())) as pool:
        headers = list(tqdm(pool.imap(get_header_info, urls), total=len(urls), desc=f"Batch {batch_idx + 1}"))
    batch_df["header"] = headers
    batch_df["text"] = batch_df["url"] + " [SEP] " + batch_df["header"]

    # 중간 저장
    part_path = os.path.join(batch_dir, f"intermediate_output_part_{batch_idx + 1}.csv")
    batch_df[["text", "label"]].to_csv(part_path, index=False)

print("모든 배치 완료. 병합 중...")

# 병합
all_batches = []
for batch_idx in range(num_batches):
    part_file = os.path.join(batch_dir, f"intermediate_output_part_{batch_idx + 1}.csv")
    part_df = pd.read_csv(part_file)
    all_batches.append(part_df)

final_df = pd.concat(all_batches, ignore_index=True)
final_df.to_csv(final_output_file, index=False)

print(f"병합 완료: {final_output_file} 저장됨")
