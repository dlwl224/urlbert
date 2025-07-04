import pandas as pd

df = pd.read_csv("urlbert_input.csv")  # 현재 작업 디렉토리에 이 파일이 있는 경우

# NOHEADER 비율 계산
total = len(df)
noheader_count = df['text'].str.contains("NOHEADER").sum()
ratio = (noheader_count / total) * 100

print(f"전체: {total}개")
print(f"NOHEADER 개수: {noheader_count}개")
print(f"NOHEADER 비율: {ratio:.2f}%")
