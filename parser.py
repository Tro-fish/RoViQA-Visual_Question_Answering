import argparse

def train_parse_args():
    parser = argparse.ArgumentParser(description="Train VQA Model")
    parser.add_argument('--train_annotation_path', type=str, default='/home/wani/Desktop/Temp/Visual-Question-Answering/datasets/dacon/train.csv', help="Path to the training data CSV file")
    parser.add_argument('--test_annotation_path', type=str, default='/home/wani/Desktop/Temp/Visual-Question-Answering/datasets/dacon/test.csv', help="Path to the test data CSV file")
    parser.add_argument('--train_img_path', type=str, default='/home/wani/Desktop/Temp/Visual-Question-Answering/datasets/dacon/image/train', help="Path to the training images directory")
    parser.add_argument('--test_img_path', type=str, default='/home/wani/Desktop/Temp/Visual-Question-Answering/datasets/dacon/image/test', help="Path to the test images directory")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=112, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
    return parser.parse_args()

def test_parser_args():
    parser = argparse.ArgumentParser(description="Inference VQA Model")
    parser.add_argument('--vqa_model_path', type=str, default='models/best_vqa_model.pth', help="Path to the trained model file")
    parser.add_argument('--image_path', type=str, default='/home/wani/Desktop/Visual-Question-Answering/images/apple_image.jpg', help="Path to the image file")
    parser.add_argument('--annotation_text', type=str, default="What is the object in this picture?", help="Annotation text for the image")
    return parser.parse_args()