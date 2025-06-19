import json
import os
import numpy as np
import torch
import config
from torch.utils.data import TensorDataset, DataLoader

label_to_id = {label:i for i, label in enumerate(config.ALL_LABELS)}

def load_and_process_data(file_path) -> dict[str, list[dict[str, str]]]:
    """Загрузка и обработка данных с валидацией"""
    data_path = os.path.join(config.DATA_DIR, 'prepared_data.json')
    train_data_path = os.path.join(config.DATA_DIR, 'train_data.json')
    validate_data_path = os.path.join(config.DATA_DIR, 'validate_data.json')
    if not os.path.exists(data_path):
        # Загрузка сырых данных
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip().split(' ', 1) for line in f]
        
        # Валидация данных
        processed_data = []
        for parts in lines:
            label = parts[0].split(',')[0]
            text = parts[1]
            processed_data.append({'label': label, 'text': text})
        
        # Стратифицированное разделение
        np.random.shuffle(processed_data)
        split_idx = int(0.75 * len(processed_data))
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]
        
        # Сохранение обработанных данных
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump({'train': train_data, 'val': val_data}, f, ensure_ascii=False)

        with open(train_data_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False)
        with open(validate_data_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False)
        
    with open(data_path, 'r', encoding='utf-8') as f:
        print('Data loaded succesfully')
        return json.load(f)

def create_data_loaders(data, tokenizer):
    """Создание DataLoader с токенизацией"""
    def encode(texts, labels):
        encoding = tokenizer(
            texts,  
            max_length=config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return TensorDataset(
            encoding['input_ids'],
            encoding['attention_mask'],
            torch.tensor([label_to_id[lbl] for lbl in labels])
        )
    
    # Подготовка данных
    train_dataset = encode(
        [item['text'] for item in data['train']],
        [item['label'] for item in data['train']]
    )
    print('train dataset prepared')
    val_dataset = encode(
        [item['text'] for item in data['val']],
        [item['label'] for item in data['val']]
    )
    print('validate dataset prepared')
    return (
        DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True),
        DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    )

