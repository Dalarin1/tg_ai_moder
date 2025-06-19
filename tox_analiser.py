from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-tox-detecter')
model = BertForSequenceClassification.from_pretrained('bert-tox-detecter')

def is_toxic(text:str):
    return model(tokenizer.encode(text,return_tensors='pt')).logits.tolist()[0][1]  > 1
