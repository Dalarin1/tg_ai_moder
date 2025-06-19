import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def load_model(model_dir, num_labels=1):
    # Загрузка токенизатора
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    
    # Загрузка конфига и модели
    model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return model, tokenizer