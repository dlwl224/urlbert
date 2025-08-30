import pandas as pd

# 파일 경로
fin_path = "/home/kong/urlbert/url_bert/urlbert2/dataset/fin_input.csv"
urlbert_path = "/home/kong/urlbert/url_bert/urlbert2/dataset/urlbert_input.csv"
merged_path = "/home/kong/urlbert/url_bert/urlbert2/dataset/merged.csv"

# CSV 파일 읽기
df_fin = pd.read_csv(fin_path)
df_urlbert = pd.read_csv(urlbert_path)

# 데이터 결합
merged_df = pd.concat([df_urlbert, df_fin], ignore_index=True)

# 저장
merged_df.to_csv(merged_path, index=False)
print(f"✅ 병합 완료: {merged_path}")
