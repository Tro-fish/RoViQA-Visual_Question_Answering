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
        image_features = self.resnet(images) # [batch_size, 1000]
        outputs = self.gpt2(question)
        output_features = outputs.last_hidden_state # [batch_size, sequence_length, hidden_size]

        image_features = image_features.unsqueeze(1).expand(-1, output_features.size(1),-1) # [batch, sequence, 1000]

        combined = torch.cat([image_features, output_features], dim=-1) # [batch, sequence, 1000+hidden]
        output = self.classifier(combined) # [batch, vocab_size]
        return output
    
    def train_model(self, loader, optimizer, criterion, device, num_epochs=1):
        for epoch in range(num_epochs):
            self.train()  # 모델을 학습 모드로 전환
            total_loss = 0
            batch_losses = []

            for data in tqdm(loader, total=len(loader)):
                images = data['image'].to(device)  # 이미지 데이터를 GPU로 전송
                question = data['question'].to(device)  # 질문 데이터를 GPU로 전송
                answer = data['answer'].to(device)  # 정답 데이터를 GPU로 전송

                optimizer.zero_grad()  # 옵티마이저의 기울기(그래디언트)를 초기화

                outputs = self(images, question)  # 모델의 예측값 계산

                # output: [batch_size, sequence, vocab], answer: [batch_size, sequence]
                loss = criterion(outputs.view(-1, outputs.size(-1)), answer.view(-1))  # 손실 계산
                total_loss += loss.item()  # 총 손실값에 현재 배치의 손실 추가
                batch_losses.append(loss.item())  # 배치 손실 저장

                loss.backward()  # 손실에 대한 그래디언트를 계산 (역전파)
                optimizer.step()  # 옵티마이저를 통해 파라미터 업데이트

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item()}")

            avg_loss = total_loss / len(loader)  # 평균 손실 계산
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss}")
    
    def validate_model(self, loader, device):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(loader, total=len(loader)):
                images = data['image'].to(device)
                question = data['question'].to(device)
                answer = data['answer'].to(device)

                outputs = self(images, question)
                _, predicted = torch.max(outputs, 2)

                total += answer.size(0)
                correct += (predicted == answer).sum().item()

        accuracy = correct / total
        return accuracy
    
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