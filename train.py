import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from vqa_dataset import VQADataset
from vqa_model import VQAModel
from parser import train_parse_args

args = train_parse_args()

train_annotations = pd.read_csv(args.train_annotation_path)
test_annotations = pd.read_csv(args.test_annotation_path)
sample_submission = pd.read_csv('/home/wani/Desktop/Temp/Visual-Question-Answering/datasets/dacon/sample_submission.csv')
train_img_path = args.train_img_path
test_img_path = args.test_img_path

# dataset & dataloader
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
vocab_size = len(tokenizer)
# Create a dataset
dataset = VQADataset(train_annotations, tokenizer, train_img_path, is_test=False)
batch_size = args.batch_size
# train/validation split 9:1
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# train & validation dataloader
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device is {device}")

# Model
model = VQAModel(vocab_size)
model.to(device)

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
num_epochs = args.num_epochs
total_steps = len(train_loader) * num_epochs // batch_size
scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps//10, num_training_steps=total_steps)

# Training loop
model.train_model(train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs)

# Dataset & DataLoader
test_dataset = VQADataset(test_annotations, tokenizer, test_img_path, is_test=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# inference
preds = model.inference(test_loader, device)
pad_token_id = tokenizer.pad_token_id
no_pad_output = []
for pred in preds:
    output = pred[pred != pad_token_id] # [PAD] token 제외
    no_pad_output.append(tokenizer.decode(output).strip()) # 토큰 id -> 토큰

sample_submission['answer'] = no_pad_output
sample_submission.to_csv('submission.csv', index=False)