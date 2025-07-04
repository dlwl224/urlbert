# tokenize_urls.py
from transformers import BertTokenizer
from transformers import DataCollatorForLanguageModeling
import pandas as pd
import warnings
import torch
from tqdm import tqdm
import numpy as np
import gc
import os

def read_txt(file_paths, url_data):
    """
    file_paths: ['dataset/train.txt', ...]
    url_data:    빈 리스트
    """
    for file in file_paths:
        # <label>\t<URL> 형식 읽기
        df = pd.read_csv(
            file,
            sep='\t',
            header=None,
            names=['label','url'],
            encoding='utf-8'
        )
        url_data.append(df['url'])
    return url_data

if __name__ == "__main__":
    # 1) 토크나이저 & MLM collator
    bert_tokenizer = BertTokenizer(vocab_file="./bert_tokenizer/vocab.txt")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bert_tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # 2) 학습용 데이터(.txt) 경로 지정
    file_paths = ["dataset/train.txt"]
    url_data   = []
    train_series = read_txt(file_paths, url_data)
    train_data   = pd.concat(train_series, ignore_index=True).to_numpy()

    # 3) 검증용 데이터(.txt) 경로 지정
    filename = "dataset/test.txt"
    df_val   = pd.read_csv(
        filename,
        sep='\t',
        header=None,
        names=['label','url'],
        encoding='utf-8'
    )
    val_data = df_val['url'].values

    warnings.filterwarnings('ignore')

    # 4) 청크 크기 설정
    max_num     = 200000
    max_num_val = 40000

    # 5) 저장 폴더 생성
    os.makedirs("./tokenized_data/train", exist_ok=True)
    os.makedirs("./tokenized_data/val",   exist_ok=True)

    # 6) 학습 데이터 토크나이즈 & 저장
    for i, chunk in enumerate(tqdm(
        np.array_split(train_data, max(1, len(train_data)/max_num)),
        desc="Split Train_Data and Tokenize"
    )):
        input_ids_train       = []
        token_type_ids_train  = []
        attention_masks_train = []
        labels_train          = []

        for sent in tqdm(chunk, desc="Tokenizing Train_Data"):
            enc    = bert_tokenizer.encode_plus(
                        sent,
                        add_special_tokens=True,
                        max_length=64,
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                     )
            masked_ids, masked_labels = data_collator.torch_mask_tokens(enc["input_ids"])
            input_ids_train.append(masked_ids)
            token_type_ids_train.append(enc["token_type_ids"])
            attention_masks_train.append(enc["attention_mask"])
            labels_train.append(masked_labels)

        torch.save(input_ids_train,
                   f"./tokenized_data/train/train_input_ids_{i}.pt")
        torch.save(token_type_ids_train,
                   f"./tokenized_data/train/train_token_type_ids_{i}.pt")
        torch.save(attention_masks_train,
                   f"./tokenized_data/train/train_attention_mask_{i}.pt")
        torch.save(labels_train,
                   f"./tokenized_data/train/train_labels_{i}.pt")

        del chunk
        gc.collect()

    # 7) 검증 데이터 토크나이즈 & 저장
    for i, chunk in enumerate(tqdm(
        np.array_split(val_data, max(1, len(val_data)/max_num_val)),
        desc="Split Val_Data and Tokenize"
    )):
        input_ids_val       = []
        token_type_ids_val  = []
        attention_masks_val = []
        labels_val          = []

        for sent in tqdm(chunk, desc="Tokenizing Val_Data"):
            enc    = bert_tokenizer.encode_plus(
                        sent,
                        add_special_tokens=True,
                        max_length=64,
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                     )
            masked_ids, masked_labels = data_collator.torch_mask_tokens(enc["input_ids"])
            input_ids_val.append(masked_ids)
            token_type_ids_val.append(enc["token_type_ids"])
            attention_masks_val.append(enc["attention_mask"])
            labels_val.append(masked_labels)

        torch.save(input_ids_val,
                   f"./tokenized_data/val/val_input_ids_{i}.pt")
        torch.save(token_type_ids_val,
                   f"./tokenized_data/val/val_token_type_ids_{i}.pt")
        torch.save(attention_masks_val,
                   f"./tokenized_data/val/val_attention_mask_{i}.pt")
        torch.save(labels_val,
                   f"./tokenized_data/val/val_labels_{i}.pt")

        del chunk
        gc.collect()

