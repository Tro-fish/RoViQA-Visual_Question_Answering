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
from transformers import get_cosine_schedule_with_warmup
from vqa_dataset import VQADataset
from vqa_model import VQAModel
from tqdm.auto import tqdm

# 데이터 준비
train_df = pd.read_csv('datasets/dacon/train.csv')
test_df = pd.read_csv('datasets/dacon/test.csv')
sample_submission = pd.read_csv('datasets/dacon/sample_submission.csv')
train_img_path = 'datasets/dacon/image/train'
test_img_path = 'datasets/dacon/image/test'

# Dataset & DataLoader 준비
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = len(tokenizer)

# Dataset 생성
dataset = VQADataset(train_df, tokenizer, train_img_path, is_test=False)

# Train/Validation Split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Device 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Current device is {device}")

# Model 생성
model = VQAModel(vocab_size)
model.to(device)

# Criterion, Optimizer, Scheduler 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
num_epochs = 5
total_steps = len(train_loader) * num_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training 함수 수정
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, total=len(train_loader)):
            images = data['image'].to(device)
            question = data['question'].to(device)
            answer = data['answer'].to(device)

            optimizer.zero_grad()

            outputs = model(images, question)
            loss = criterion(outputs.view(-1, outputs.size(-1)), answer.view(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item()}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss}")

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader, total=len(val_loader)):
                images = data['image'].to(device)
                question = data['question'].to(device)
                answer = data['answer'].to(device)

                outputs = model(images, question)
                loss = criterion(outputs.view(-1, outputs.size(-1)), answer.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss}")

# 학습 실행
train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs)

# Dataset & DataLoader 생성 (테스트 데이터)
test_dataset = VQADataset(test_df, tokenizer, test_img_path, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inference 실행
preds = model.inference(test_loader, device)
pad_token_id = tokenizer.pad_token_id
no_pad_output = []
for pred in preds:
    output = pred[pred != pad_token_id]  # [PAD] token 제외
    no_pad_output.append(tokenizer.decode(output).strip())  # 토큰 id -> 토큰

sample_submission['answer'] = no_pad_output
sample_submission.to_csv('submission.csv', index=False)