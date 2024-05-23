import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models # 이미지
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import GPT2Tokenizer, GPT2Model # 텍스트
from tqdm.auto import tqdm
   

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = len(tokenizer)

gpt2 = GPT2Model.from_pretrained('gpt2')
gpt2.resize_token_embeddings(vocab_size)

# 예제 질문 토큰화
question = tokenizer.encode_plus(
    "What is the capital of France?",
    return_tensors='pt',
    padding='max_length',
    max_length=32,
    truncation=True
)
print("question: ",question)

# GPT-2를 통한 특징 추출
outputs = gpt2(question['input_ids'])  # [batch_size, sequence_length, hidden_size]
#print("outputs: ",outputs)
output_features = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
print(output_features.shape)
# 출력: torch.Size([batch_size, sequence_length, hidden_size])

"""
# 예제 이미지 텐서 생성
batch_size = 4
images = torch.randn(batch_size, 3, 224, 224)  # [batch_size, 3, 224, 224]
resnet = models.resnet50(pretrained=True)
# ResNet50을 통한 특징 추출 및 크기 재구성
image_features = resnet(images)  # [batch_size, 1000]
print(image_features.shape)
print(image_features)
image_features = image_features.view(image_features.size(0), -1)  # [batch_size, 1000]
print(image_features.shape)
print(image_features)
"""