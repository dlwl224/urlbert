# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
import pandas as pd
import numpy as np
from torch.utils.data import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import time

# %%
def dataPreprocessFromCSV(filename, input_ids, input_types, input_masks, label):
    """
    Preprocess data from a CSV file containing URLs and labels.

    :param filename: The path to the CSV data file.
    :param input_ids: List to store input char IDs.
    :param input_types: List to store segment IDs.
    :param input_masks: List to store attention masks.
    :param label: List to store labels.
    :return: None
    """
    pad_size = 200
    tokenizer = BertTokenizer("./bert_tokenizer/vocab.txt")  # Initialize the tokenizer

    data = pd.read_csv(filename, encoding='utf-8')
    for i, row in tqdm(data.iterrows(), total=len(data)):
        x1 = row['url']  # Replace with the column name in your CSV file where the text data is located
        x1 = tokenizer.tokenize(x1)
        tokens = ["[CLS]"] + x1 + ["[SEP]"]

        # Get input_id, seg_id, att_mask
        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(ids))
        masks = [1] * len(ids)

        # Pad if short, truncate if long
        if len(ids) < pad_size:
            types = types + [1] * (pad_size - len(ids))  # Set segment to 1 for the masked part
            masks = masks + [0] * (pad_size - len(ids))
            ids = ids + [0] * (pad_size - len(ids))
        else:
            types = types[:pad_size]
            masks = masks[:pad_size]
            ids = ids[:pad_size]
        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)
        assert len(ids) == len(masks) == len(types) == pad_size

        y = row['label']
        if y == 'malicious':
            label.append([1])
        elif y == 'benign':
            label.append([0])
        elif y == "1":
            label.append([1])
        elif y == "0":
            label.append([0])

# %%
input_ids = []  # input char ids
input_types = []  # segment ids
input_masks = []  # attention mask
label = []

dataPreprocessFromCSV("~/home/sqanar_u/myproject/sQanAR/url_bert/urlbert2/dataset/train.csv", input_ids, input_types, input_masks, label)

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE

# %%
def spiltDatast_bert(input_ids, input_types, input_masks, label):
    """
    Split the dataset into training and testing sets.

    :param input_ids: List of input character IDs.
    :param input_types: List of segment IDs.
    :param input_masks: List of attention masks.
    :param label: List of labels.
    :return: Split datasets for training and testing.
    """
    # Randomly shuffle the indices
    random_order = list(range(len(input_ids)))
    np.random.seed(2024)  # Fix the seed
    np.random.shuffle(random_order)
    print(random_order[:10])

    # Split the dataset into 80% training and 20% testing
    input_ids_train = np.array([input_ids[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    input_types_train = np.array([input_types[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    input_masks_train = np.array([input_masks[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    y_train = np.array([label[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    print("input_ids_train.shape:" + str(input_ids_train.shape))
    print("input_types_train.shape:" + str(input_types_train.shape))
    print("input_masks_train.shape:" + str(input_masks_train.shape))
    print("y_train.shape:" + str(y_train.shape))

    input_ids_test = np.array([input_ids[i] for i in random_order[int(len(input_ids) * 0.8):int(len(input_ids) * 1)]])
    input_types_test = np.array([input_types[i] for i in random_order[int(len(input_ids) * 0.8):int(len(input_ids) * 1)]])
    input_masks_test = np.array([input_masks[i] for i in random_order[int(len(input_ids) * 0.8):int(len(input_ids) * 1)]])
    y_test = np.array([label[i] for i in random_order[int(len(input_ids) * 0.8):int(len(input_ids) * 1)]])
    print("input_ids_test.shape:" + str(input_ids_test.shape))
    print("input_types_test.shape:" + str(input_types_test.shape))
    print("input_masks_test.shape:" + str(input_masks_test.shape))
    print("y_test.shape:" + str(y_test.shape))

    return input_ids_train, input_types_train, input_masks_train, y_train, input_ids_test, input_types_test, input_masks_test, y_test

# %%
print(len(input_ids))
print(len(label))

# %%
input_ids_train, input_types_train, input_masks_train, y_train, input_ids_val, input_types_val, input_masks_val, y_val = spiltDatast_bert(
        input_ids, input_types, input_masks, label)
BATCH_SIZE = 64
train_data = TensorDataset(torch.tensor(input_ids_train).to(DEVICE),
                               torch.tensor(input_types_train).to(DEVICE),
                               torch.tensor(input_masks_train).to(DEVICE),
                               torch.tensor(y_train).to(DEVICE))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

val_data = TensorDataset(torch.tensor(input_ids_val).to(DEVICE),
                              torch.tensor(input_types_val).to(DEVICE),
                              torch.tensor(input_masks_val).to(DEVICE),
                              torch.tensor(y_val).to(DEVICE))
val_sampler = SequentialSampler(val_data)
val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

# %%
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
)

config_kwargs = {
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
    "hidden_dropout_prob": 0.1,
    "vocab_size": 5000,
}

config = AutoConfig.from_pretrained("/home/sqanar_u/myproject/sQanAR/url_bert/urlbert2/bert_config", **config_kwargs)
print(config)

bert_model = AutoModelForMaskedLM.from_config(
    config=config,
)
bert_model.resize_token_embeddings(config_kwargs["vocab_size"])
print(bert_model)

# %%
bert_dict = torch.load("/home/sqanar_u/myproject/sQanAR/url_bert/urlbert2/bert_model/urlbert (1).pt")
bert_model.load_state_dict(bert_dict)

# %%
class BertForSequenceClassification(nn.Module):
    def __init__(self, bert, freeze=False):
        super(BertForSequenceClassification, self).__init__()
        self.bert = bert
        for name, param in self.bert.named_parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]
        outputs = self.bert(context, attention_mask=mask, token_type_ids=types, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:,0,:]
        out = self.dropout(hidden_states)
        out = self.classifier(out)
        
        return out

# %%
model = BertForSequenceClassification(bert_model)
model.bert.cls = nn.Sequential()
model.to(DEVICE)

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

# %%
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        start_time = time.time()
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        
        y_pred = model([x1, x2, x3])
        model.zero_grad()
        
        loss = F.cross_entropy(y_pred, y.squeeze())
        loss.backward()
        
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\t Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))

# %%
def validation(model, device, test_loader):
    """
    Perform model validation on the test data.

    :param model: The model to be validated.
    :param device: The device to run validation on (e.g., CPU or GPU).
    :param test_loader: The data loader for test data.
    :return: A tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model([x1, x2, x3])

        test_loss += F.cross_entropy(y_, y.squeeze()).item()

        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2 outputs, representing the maximum value and its index

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'],
                yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the confusion matrix plot
    plt.savefig('confusion_matrix.png')

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

# %%
torch.cuda.empty_cache()

# %%
best_acc = 0.0
NUM_EPOCHS = 5
PATH = '/hy-tmp/modelx_URLBERT_80.pth'
for epoch in range(1, NUM_EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    acc, precision, recall, f1 = validation(model, DEVICE, val_loader)
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), PATH)
    print("acc is: {:.4f}, best acc is {:.4f}".format(acc, best_acc))


