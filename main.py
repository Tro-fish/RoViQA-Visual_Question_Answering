import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models # 이미지
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer # 텍스트
from vqa_dataset import VQADataset
from vqa_model import VQAModel
from tqdm.auto import tqdm
   

train_df = pd.read_csv('/home/wani/Desktop/Temp/Visual-Question-Answering/datasets/dacon/train.csv')
test_df = pd.read_csv('/home/wani/Desktop/Temp/Visual-Question-Answering/datasets/dacon/test.csv')
sample_submission = pd.read_csv('/home/wani/Desktop/Temp/Visual-Question-Answering/datasets/dacon/sample_submission.csv')
train_img_path = '/home/wani/Desktop/Temp/Visual-Question-Answering/datasets/dacon/image/train'
test_img_path = '/home/wani/Desktop/Temp/Visual-Question-Answering/datasets/dacon/image/test'

# dataset & dataloader
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
vocab_size = len(tokenizer)
# dataset 생성
dataset = VQADataset(train_df, tokenizer, train_img_path, is_test=False)

# train/validation split 9:1
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# train & validation dataloader
train_loader = DataLoader(train_dataset, batch_size=112, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=112, shuffle=False)

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device is {device}")

# Model
model = VQAModel(vocab_size)
# google-bert/bert-base-uncased --> 110M
# google/vit-base-patch16-224 --> 86M
# ourt vqa_model --> 205M

# 학습된 모델 불러오기
# model.vit.load_state_dict(torch.load("best_image_model.pth", map_location=device))
# model.bert.load_state_dict(torch.load("best_text_model.pth", map_location=device))

model.to(device)

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# Training loop
model.train_model(train_loader, val_loader, optimizer, criterion, device, num_epochs=5)

# Dataset & DataLoader
test_dataset = VQADataset(test_df, tokenizer, test_img_path, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=112, shuffle=False)

# inference
preds = model.inference(test_loader, device)
pad_token_id = tokenizer.pad_token_id
no_pad_output = []
for pred in preds:
    output = pred[pred != pad_token_id] # [PAD] token 제외
    no_pad_output.append(tokenizer.decode(output).strip()) # 토큰 id -> 토큰

sample_submission['answer'] = no_pad_output
sample_submission.to_csv('submission.csv', index=False)
solution = pd.read_csv('solution.csv')