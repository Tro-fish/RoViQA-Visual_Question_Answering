import json
import pandas as pd

with open('datasets/vqa_train_questions.json') as f:
    question_data = json.load(f)
with open('datasets/vqa_train_annotations.json') as f:
    answer_data = json.load(f)

print((question_data['questions'][:3]))
print()
print((answer_data['annotations'][0]))