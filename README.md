내가 한 것 
fintune -> phishing


modelx_URLBERT_80.pth :파인튜닝된 모델




| 범주                  | 폴더·파일                                            | 핵심 기능                                                                                                                                                                                                  |
| ------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **사전에 학습‑돼 있는 리소스** | `bert_config/`, `bert_model/`, `bert_tokenizer/` | ▸ `bert_config/` : `BertForMaskedLM` 구성<br>▸ `bert_tokenizer/` : URL 전용 vocab (`vocab.txt` 등)<br>▸ `bert_model/` : 논문에서 미리 학습한 URLBERT 가중치 (`urlBERT.pt`)                        |
| **전처리**             | `tokenize.py`                                    | 원시 URL을 토큰 ID 시퀀스로 변환 후 `./tokenized_data/…`에 `.pt` 파일로 저장                                                                                                                                             |
| **모델 정의·빌드**        | `buildmodel.py`                                  | `AutoConfig` → `AutoModelForMaskedLM` 인스턴스화. vocab 크기를 파라미터로 받아 임베딩 재조정                                                                                              |
| **데이터 부로더**         | `dataloader.py`                                  | (DDP 포함) 학습·검증 `DataLoader` 생성                                                                                                                                        |
| **사전학습 파이프라인**      | `main_stage_1.py`, `main_stage_2.py`             | Stage 1: MLM + STD(Shuffle‑Token Detection) + RTD(Replaced‑Token Detection) 학습<br>Stage 2: 추가 epoch 또는 더 큰 배치 등으로 파인 사전학습. 두 파일 모두 여러 GPU(DistributedDataParallel) 전제            |
| **하이퍼파라미터**         | `options.py`                                     | `--epochs --batch_size --lr --weight_decay --local_rank` 등 CLI 인자 정의                                                                                               |
| **실험·파인튜닝**         | `finetune/…` 하위                                  | `phishing/`, `advertising/`, `web_classification/`, `Multi‑Task/` 등 태스크별 스크립트<br>                                    |
| **보조 유틸**           | `AL.py`, `DropAL.py`, `timerecord.py` 등          | 활성 학습(Active Learning) 실험, 시간 로깅 등                                                                                                                                                                     |

