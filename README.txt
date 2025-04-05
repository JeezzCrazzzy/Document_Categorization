# Distillation with DistilBERT

## Project Description
This project implements knowledge distillation using DistilBERT and LayoutLMv3 for token classification. It focuses on distilling knowledge from a large teacher model (LayoutLMv3) into a smaller and more efficient student model (DistilBERT). The goal is to improve the efficiency of token classification tasks on structured documents while maintaining high accuracy.

### Key Features
- Uses LayoutLMv3 as the teacher model and DistilBERT as the student model.
- Leverages the FUNSD dataset for form understanding.
- Implements knowledge distillation with KL divergence loss.
- Fine-tunes the teacher model before transferring knowledge.
- Evaluates model performance using precision, recall, F1-score, and accuracy.

---

## Installation
To run this project, install the required dependencies using:
```bash
pip install datasets evaluate seqeval torch transformers
```

---

## Usage

### 1. Load Dependencies
```python
import torch
from transformers import AutoProcessor, AutoTokenizer, DistilBertTokenizer
```

### 2. Set Up Models
```python
teacher_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base").to(device)
student_model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels).to(device)
```

### 3. Prepare Dataset
```python
dataset = load_dataset("nielsr/funsd-layoutlmv3")
```

### 4. Train the Teacher Model
```python
trainer.train()
```

### 5. Distill Knowledge to Student Model
```python
train_student_model(student_model, teacher_model, train_dataloader, optimizer, num_epochs=10)
```

### 6. Evaluate the Student Model
```python
evaluation_results = evaluate_student_model(student_model, eval_dataloader, label_list)
print(evaluation_results)
```

---

## Model Training Pipeline
1. **Preprocess the dataset**: Tokenize input text and images.
2. **Train the teacher model**: Fine-tune LayoutLMv3.
3. **Prepare student dataset**: Remove image-based features.
4. **Train student model using distillation**: Optimize with a weighted combination of soft and hard losses.
5. **Evaluate performance**: Measure precision, recall, F1-score, and accuracy.

---

## Results
The student model achieves a balance between efficiency and accuracy compared to the teacher model. The final evaluation metrics include:
- Precision: **41.67%**
- Recall: **2.27%**
- F1-score: **4.32%**
- Accuracy: **21.17%**

(Note: Replace X, Y, Z, and W with actual evaluation results.)

---

## Upcoming Iterations
The student model scores poor on the evaluation standards as compared to the teacher model(75%) accuracy, the next iterations would include the following changes:
1. Using a bigger dataset for training the model. "|https://huggingface.co/datasets/functionX86/layoutlmv3-cordv2-mapped/viewer/default/train?row=0&views%5B%5D=train" : this would be the future dataset for finetuning teacher model and training.
2. If this doesn't improve the accuracy of the model, I will change the student model and use TinyBERT or MobileBERT.

---

## Acknowledgments
This project is based on Hugging Face Transformers and datasets from FUNSD.

---

## License
This project is licensed under the MIT License.

