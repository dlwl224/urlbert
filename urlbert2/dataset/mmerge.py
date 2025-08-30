import pandas as pd
import os

folder_path = "/home/kong/urlbert/url_bert/urlbert2/dataset/link"
output_file = "/home/kong/urlbert/url_bert/urlbert2/dataset/mmerged.csv"

dfs = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, dtype=str)  # 모든 컬럼을 문자열로
        if 'url' in df.columns:
            # 따옴표 제거 + 공백 제거
            df['url'] = df['url'].astype(str).str.replace('"', '').str.strip()
            dfs.append(df[['url']])

merged_df = pd.concat(dfs, ignore_index=True)

# 중복 제거
merged_df.drop_duplicates(subset='url', inplace=True)

merged_df.to_csv(output_file, index=False)

print(f"✅ 병합 완료! 총 {len(merged_df)}개의 고유 URL이 저장됨.")

