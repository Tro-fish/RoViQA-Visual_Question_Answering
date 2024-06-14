<div align="center">

# RoViQA<br>(Open-ended) Visual-Question Answering Model

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
<br><b> RoViQA: Visual-Question-Answering model created by combining Roberta and ViT</b> <br><br>
This repository contains the code for **RoViQA**, a Visual Question Answering (VQA) model that combines image features extracted using Vision Transformer (ViT) and text features extracted using RoBERTa. The project includes training, inference, and various utility scripts.
</div>

## Model Architecture
<p align="center">
  <img src="https://github.com/Tro-fish/Visual-Question-Answering/assets/79634774/f7d0eb20-f3b4-4f69-880d-d412ed32ab68" alt="Description of the image" width="100%" />
</p>

## RoViQA Overview
RoViQA is a Visual Question Answering (VQA) model that leverages the power of Vision Transformer (ViT) and RoBERTa to understand and answer questions about images. By combining the strengths of these two models, RoViQA can effectively process and interpret both visual and textual information to provide accurate answers.

## Model parameter
- Base Models
	- Roberta-base: 110M parameters
	- ViT-base: 86M parameters
- **RoViQA**: 215M parameters

## Repository Structure

- **vqa_model.py**: Definition of the VQA model architecture.
- **vqa_dataset.py**: Script to handle the dataset loading and preprocessing.
- **train.py**: Script to train the VQA model.
- **inference.py**: Script to perform inference using the trained VQA model.
- **parser.py**: Utility script to parse command-line arguments for training and inference.

## Install
### Setup `python` environment
```
conda create -n VQA python=3.8
```
### Install other dependencies
```
pip install -r requirements.txt
```
## Dataset

- The dataset used for training is based on the [VQA Consortium dataset](https://visualqa.org). It consists of pairs of images and questions, totaling 107,231 pairs. The dataset was used to train the model for 5 epochs.
- The dataset I used can be downloaded from [this](https://dacon.io/competitions/official/236118/data) link

## Scripts
### Train RoViQA
```
python train.py --train_annotation_path your_annotation_path.csv --test_annotation_path your_annotation_path.csv --train_img_path your_dataset_path.csv --test_img_path your_dataset_path.csv --lr 5e-5 --batch_size 112 --num_epochs 5 --weight_decay 0.01
```
Hyperparameters

	•	Learning rate: 5e-5
	•	Batch size: 112
	•	Weight decay: 0.01
	•	LR Scheduler: Cosine
 
Machine Environment

	•	CPU: i7-11700K
	•	GPU: RTX 3090
	•	RAM: 64GB
 
### Inference RoViQA
- **To perform inference using the trained model, use the inference.py script. You need to specify the model path, image path, and the annotation text**
- I've put a test image in the images folder, so you're good to go.
- I've uploaded my trained model to [HuggingFace](https://huggingface.co/Trofish/RoViQA), ready for you to use.
```
python inference.py —image_model_path image_model.pth —text_model_path text_model.pth —image_path your_image.jpg —annotation_text "Your question"
```
