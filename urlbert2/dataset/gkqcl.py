import pandas as pd

# 파일 경로 설정
csv_file_1 = "/home/kong/urlbert/url_bert/urlbert2/dataset/urlbert_benign.csv"
csv_file_2 = "/home/kong/urlbert/url_bert/urlbert2/dataset/urlbert_phishing.csv"
output_file = "/home/kong/urlbert/url_bert/urlbert2/dataset/urlbert_final.csv"

# CSV 파일 읽기
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

# 병합
merged_df = pd.concat([df1, df2], ignore_index=True)

# 저장
merged_df.to_csv(output_file, index=False)

print(f"✅ 병합 완료! 총 {len(merged_df)}개의 URL이 저장됨.")
