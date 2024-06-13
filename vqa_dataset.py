
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

class VQADataset(Dataset):
    def __init__(self, df, tokenizer, img_path, is_test=False):
        self.df = df # train.csv의 데이터
        self.tokenizer = tokenizer # TextEncoder의 tokenizer
        self.transform = TRANSFORM
        self.img_path = img_path
        self.is_test = is_test

    def __len__(self): # 데이터의 길이 return
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = os.path.join(self.img_path, row['image_id'] + '.jpg') # 각 이미지 파일의 경로
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        question = row['question'] # 질문
        question = self.tokenizer.encode_plus(
            question,
            truncation=True,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        if not self.is_test:
            answer = row['answer'] # 답변
            answer = self.tokenizer.encode_plus(
                answer,
                max_length=32,
                padding='max_length',
                truncation=True,
                return_tensors='pt')
            
            return {
                'image': image.squeeze(),
                'question': {key: val.squeeze(0) for key, val in question.items()},
                'answer': answer['input_ids'].squeeze()
            }
        
        else:
            return {
                'image': image,
                'question': {key: val.squeeze(0) for key, val in question.items()},
            }