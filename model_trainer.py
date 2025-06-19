import torch 
import config
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from model_loader import load_model
from torch.utils.data import DataLoader
from data_processor import load_and_process_data, create_data_loaders
from datasets import load_dataset
    
model, tokenizer = load_model(config.MODEL_DIRECTORY, num_labels=len(config.ALL_LABELS))

def preprocess_function(examples):
    # Токенизация с обрезанием и дополнением
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    # Преобразование текстовых меток в числовые
    tokenized["labels"] = [config.LABEL_TO_ID[label] for label in examples["label"]]
    return tokenized


model.train()
data = load_and_process_data('train data\\dataset_labeled.txt')
dataset = load_dataset('json', data_files='train data\\train_data.json')
dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["text", "label"]  # Удаляем исходные текстовые поля
)
val_dataset = load_dataset('json', data_files='train data\\validate_data.json')
val_dataset = val_dataset.map(
        preprocess_function,
    batched=True,
    remove_columns=["text", "label"]  # Удаляем исходные текстовые поля
)
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs",
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=False,
)
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].take(10000),
    eval_dataset=val_dataset['train'],  # Для примера, замените на валидационный набор
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(config.MODEL_DIRECTORY)

# 7. Сохранение модели
# model.save_pretrained(config.MODEL_DIRECTORY)
tokenizer.save_pretrained(config.MODEL_DIRECTORY)
