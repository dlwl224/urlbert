# prepare_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split

# 1) CSV 로드
df = pd.read_csv('dataset/bert_data.csv')           # bert_data.csv가 루트에 있어야 함

# 2) 레이블 필터링 및 매핑 (0→benign, 2→malicious)
df = df[df['label'].isin([0,2])]            # 혹시 다른 레이블이 있으면 제거
df['label'] = df['label'].map({0:'benign', 2:'malicious'})

# 3) train/test 분할 (80/20)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# 4) 텍스트 파일로 저장 (<label>\t<URL> 형식)
for split_name, split_df in [('train', train_df), ('test', test_df)]:
    lines = split_df['label'] + '\t' + split_df['url']
    with open(f'dataset/{split_name}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

print("✅ dataset/train.txt, dataset/test.txt 생성 완료")
