import torch
import torch.nn as nn
from transformers import RobertaModel, ViTModel
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
        
        self.bert = RobertaModel.from_pretrained('FacebookAI/roberta-base')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fcnn = FCNN(self.bert.config.hidden_size, [1024, 512, 256], vocab_size)

    def forward(self, images, question):
        # Extracting CLS tokens from ViT
        vit_outputs = self.vit(images)
        cls_token = vit_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Processing question sequences in BERT
        outputs = self.bert(**question)
        question_features = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]

        # Multiply CLS tokens element-wise by all question sequences
        combined = cls_token.unsqueeze(1) * question_features  # [batch_size, sequence_length, hidden_size]

        # Final output via FCNN
        output = self.fcnn(combined)  # [batch_size, sequence_length, vocab_size]
        
        return output
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, vocab_size, device):
        model = cls(vocab_size)
        state_dict = torch.hub.load_state_dict_from_url(f'https://huggingface.co/{model_name_or_path}/resolve/main/RoViQA.pth', map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    
    def train_model(self, loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=1):
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            self.train()  
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
                # save model
                torch.save(self.state_dict(), "best_vqa_model.pth")
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
