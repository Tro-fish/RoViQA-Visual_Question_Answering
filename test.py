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
from vqa_dataset import VQADataset
from vqa_model import VQAModel
from tqdm.auto import tqdm
   

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