import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
from tqdm import tqdm

class FCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Activation function
        self.activation = nn.ReLU()
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        # Output layer without activation
        x = self.layers[-1](x)
        return x

class VQAModel(nn.Module):
    def __init__(self, vocab_size):
        super(VQAModel, self).__init__()
        
        self.bert = BertModel.from_pretrained('google-bert/bert-base-uncased')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fcnn = FCNN(self.bert.config.hidden_size, [512, 256], vocab_size)

    def forward(self, images, question):
        # ViT에서 CLS 토큰 추출
        vit_outputs = self.vit(images)
        cls_token = vit_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # BERT에서 질문 시퀀스 처리
        outputs = self.bert(**question)
        question_features = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]

        # CLS 토큰을 모든 질문 시퀀스에 element-wise 곱
        combined = cls_token.unsqueeze(1) * question_features  # [batch_size, sequence_length, hidden_size]

        # FCNN을 통한 최종 출력
        output = self.fcnn(combined)  # [batch_size, sequence_length, vocab_size]
        
        return output
    
    def train_model(self, loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=1):
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            self.train()  # 모델을 학습 모드로 전환
            total_loss = 0
            batch_losses = []

            for data in tqdm(loader, total=len(loader)):
                images = data['image'].to(device)
                question = {key: val.to(device) for key, val in data['question'].items()}
                answer = data['answer'].to(device)

                optimizer.zero_grad()

                outputs = self(images, question)

                loss = criterion(outputs.view(-1, outputs.size(-1)), answer.view(-1))
                total_loss += loss.item()
                batch_losses.append(loss.item())

                loss.backward()
                optimizer.step()
                scheduler.step()

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item()}")

            avg_loss = total_loss / len(loader)  
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss}")

            val_accuracy = self.validate_model(val_loader, device)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.vit.state_dict(), "best_image_model.pth")
                torch.save(self.bert.state_dict(), "best_text_model.pth")
                print(f"New best model saved with accuracy: {val_accuracy:.4f}")

    def validate_model(self, loader, device):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(loader, total=len(loader)):
                images = data['image'].to(device)
                question = {key: val.to(device) for key, val in data['question'].items()}
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
                question = {key: val.to(device) for key, val in data['question'].items()}

                outputs = self(images, question)

                _, pred = torch.max(outputs, dim=2)
                preds.extend(pred.cpu().numpy())

        return preds
