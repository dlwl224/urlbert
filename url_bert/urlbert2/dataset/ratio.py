import pandas as pd

# train/test CSV 경로
TRAIN_CSV = "/home/kong/urlbert/url_bert/urlbert2/dataset/urlbert_input.csv"
TEST_CSV  = "/home/kong/urlbert/url_bert/urlbert2/dataset/urlbert_input_test.csv"

for name, path in [("Train", TRAIN_CSV), ("Test", TEST_CSV)]:
    df = pd.read_csv(path)
    counts = df["label"].value_counts()
    freqs = df["label"].value_counts(normalize=True) * 100
    print(f"\n=== {name} set ===")
    print(counts.to_string())
    print((freqs.round(2)).to_string() + " (%)")
