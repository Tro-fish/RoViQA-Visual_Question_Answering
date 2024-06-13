<div align="center">

# (Open-ended) Visual-Question Answering

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
<br><b> Visual-Question-Answering model created by combining Roberta and ViT</b> <br>
This repository contains the code for a Visual Question Answering (VQA) model. The model combines image features and text features extracted using ViT and RoBERTa. The project includes training, inference, and various utility scripts.
</div>

## Model Architecture
<p align="center">
  <img src="https://github.com/Tro-fish/Visual-Question-Answering/assets/79634774/f7d0eb20-f3b4-4f69-880d-d412ed32ab68" alt="Description of the image" width="100%" />
</p>

## Existing Code

- **vqa_model.py**: Definition of the VQA model architecture.
- **vqa_dataset.py**: Script to handle the dataset loading and preprocessing.
- **train.py**: Script to train the VQA model.
- **inference.py**: Script to perform inference using the trained VQA model.
- **parser.py**: Utility script to parse command-line arguments for training and inference.

## Install
### Setup `python` environment
```
conda create -n VQA python=3.8   # You can also use other environment.
```
### Install other dependencies
```
pip install -r requirements.txt
```

## Scripts
### Train VQA model
```
python train.py --train_annotation_path your_annotation_path.csv --test_annotation_path your_annotation_path.csv --train_img_path your_dataset_path.csv --test_img_path your_dataset_path.csv --lr 5e-5 --batch_size 112 --num_epochs 5 --weight_decay 0.01
```
Hyperparameters

	•	Learning rate: 5e-5
	•	Batch size: 112
	•	Weight decay: 0.01
	•	LR Scheduler: Cosine
 
Machine Environment

	•	CPU: Intel i7-11700K
	•	GPU: RTX 3090
	•	RAM: 64GB
 
### Inference UDOP model
- **To perform inference using the trained model, use the inference.py script. You need to specify the model path, image path, and the annotation text**
```
python inference.py --image_model_path image_model.pth --text_model_path text_model.pth --image_path your_image.jpg --annotation_text "Your question"
```
