# Prepare training set
import os
import random
import pandas as pd

# Preprocess text data, train-test split
from fetch_clean import prep
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Model for sequence classification training
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

data = []

# Process receipts
receipts_dir = './train/receipt'

#sample 50 receipts
receipt_imgs = [filename for filename in os.listdir(receipts_dir) if filename.endswith('.jpg') or filename.endswith('.png')]
selected_receipts = random.sample(receipt_imgs, 50)

for filename in selected_receipts:
   
    text = prep(file_path= receipts_dir + filename, out_dir="./data/receipts/")
    with open(text, "r") as f:
        text = f.read()

    data.append({'text': text, 'label': 'receipt'})

# Process invoices
invoices_dir = './train/invoice'

# sample 50 invoices
invoice_imgs = [filename for filename in os.listdir(invoices_dir) if filename.endswith('.jpg') or filename.endswith('.png')]
selected_invoices = random.sample(invoice_imgs, 50)

for filename in selected_invoices:
    
    text = prep(file_path= invoices_dir + filename, out_dir="./data/invoices/")
    with open(text, "r") as f:
        text = f.read()

    data.append({'text': text, 'label': 'invoice'})

df = pd.DataFrame(data)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# Tokenize the text data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=128)

# Convert labels to numerical format
label_dict = {'receipt': 0, 'invoice': 1}
train_labels = train_labels.map(label_dict).tolist()
val_labels = val_labels.map(label_dict).tolist()

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)

class ReceiptInvoiceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReceiptInvoiceDataset(train_encodings, train_labels)
val_dataset = ReceiptInvoiceDataset(val_encodings, val_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("fine-tuned-bert-classification-model")
