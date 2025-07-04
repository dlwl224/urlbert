import pandas as pd

# 절대 경로로 직접 읽기
df = pd.read_csv(
    "/home/injeolmi/myproject/sQanAR/urlbert2-20250616T232219Z-1-001/urlbert2/dataset/test.txt",
    sep="\t",
    names=["label","url"],
    encoding="utf-8"
)

# url, label 순서로 재정렬
df = df[["url","label"]]

# 같은 디렉터리에 train.csv 로 저장
df.to_csv(
    "/home/injeolmi/myproject/sQanAR/urlbert2-20250616T232219Z-1-001/urlbert2/dataset/test.csv",
    index=False,
    encoding="utf-8"
)
