import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models # 이미지
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from transformers import GPT2Tokenizer, GPT2Model # 텍스트
from vqa_dataset import VQADataset
from vqa_model import VQAModel
from tqdm.auto import tqdm
   

train_df = pd.read_csv('datasets/dacon/train.csv')
test_df = pd.read_csv('datasets/dacon/test.csv')
sample_submission = pd.read_csv('datasets/dacon/sample_submission.csv')
train_img_path = 'datasets/dacon/image/train'
test_img_path = 'datasets/dacon/image/test'

# dataset & dataloader
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = len(tokenizer)
# dataset 생성
dataset = VQADataset(train_df, tokenizer, train_img_path, is_test=False)

# train/validation split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# dataloader 설정
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device is {device}")

# Model
model = VQAModel(vocab_size).to(device)

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(1):
    model.train_model(train_loader, optimizer, criterion, device, num_epochs=1)

    # Validation 정확도 계산
    val_accuracy = model.validate_model(val_loader, device)
    print(f"Validation Accuracy: {val_accuracy:.4f}")


# Dataset & DataLoader
test_dataset = VQADataset(test_df, tokenizer, test_img_path, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# inference
preds = model.inference(test_loader, device)

no_pad_output = []
for pred in preds:
    output = pred[pred != 50257] # [PAD] token 제외
    no_pad_output.append(tokenizer.decode(output).strip()) # 토큰 id -> 토큰

sample_submission['answer'] = no_pad_output
sample_submission.to_csv('submission.csv', index=False)
solution = pd.read_csv('solution.csv')