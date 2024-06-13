import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer
from vqa_model import VQAModel
from parser import test_parser_args

def load_model(vqa_model_path, vocab_size, device):
    model = VQAModel(vocab_size)
    model.load_state_dict(torch.load(vqa_model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def preprocess_text(annotation_text, tokenizer):
    tokens = tokenizer.encode_plus(
        annotation_text,
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    return tokens

def main():
    args = test_parser_args()
    
    # Device Settings
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Setting up the tokenizer and vocab size
    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
    vocab_size = len(tokenizer)
    
    # Load model
    model = load_model(args.vqa_model_path, vocab_size, device)
    
    # Image preprocess
    image = preprocess_image(args.image_path).to(device)
    
    # Text preprocess
    tokens = preprocess_text(args.annotation_text, tokenizer)
    tokens = {key: val.to(device) for key, val in tokens.items()}
    
    # Prediction
    with torch.no_grad():
        outputs = model(image, tokens)
        _, pred = torch.max(outputs, dim=2)
    
    # Decoding
    pred = pred.squeeze().cpu().numpy()
    pad_token_id = tokenizer.pad_token_id
    output = pred[pred != pad_token_id] # [PAD] token 제외
    result = tokenizer.decode(output)
    print("Predicted Answer:", result)

if __name__ == "__main__":
    main()