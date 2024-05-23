import torch
import torch.nn as nn
import torchvision.models as models # 이미지
from transformers import GPT2Tokenizer, GPT2Model # 텍스트
from tqdm import tqdm


class VQAModel(nn.Module):
    def __init__(self, vocab_size):
        super(VQAModel, self).__init__()
        self.vocab_size = vocab_size

        self.resnet = models.resnet50(pretrained=True)
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.resize_token_embeddings(vocab_size) # 추가한 [PAD] 토큰 반영

        combined_features_size = 1000 + self.gpt2.config.hidden_size # resnet 출력 차원 + gpt2 출력 차원
        self.classifier = nn.Linear(combined_features_size, vocab_size)

    def forward(self, images, question):
        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0),-1)

        outputs = self.gpt2(question)
        output_features = outputs.last_hidden_state # [batch, sequence, hidden]

        image_features = image_features.unsqueeze(1).expand(-1, output_features.size(1),-1) # [batch, sequence, 1000]

        combined = torch.cat([image_features, output_features], dim=-1) # [batch, sequence, 1000+hidden]
        output = self.classifier(combined) # [batch, vocab_size]
        return output
    
    def train_model(self, loader, optimizer, criterion, device):
        self.train()
        total_loss = 0

        for data in tqdm(loader, total=len(loader)):
            images = data['image'].to(device)
            question = data['question'].to(device)
            answer = data['answer'].to(device)

            optimizer.zero_grad()

            outputs = self(images, question)

            # output: [batch, sequence, vocab], answer : [batch, sequence]
            loss = criterion(outputs.view(-1, outputs.size(-1)), answer.view(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(loader)
        return avg_loss
    
    def inference(self, loader, device):
        self.eval()
        preds = []
        with torch.no_grad():
            for data in tqdm(loader, total=len(loader)):
                images = data['image'].to(device)
                question = data['question'].to(device)

                outputs = self(images, question) # [batch, sequence, vocab]

                _, pred = torch.max(outputs, dim=2) # values, indices = _, pred
                preds.extend(pred.cpu().numpy())

        return preds