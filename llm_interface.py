import torch
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Load pre-trained summarization model. In this case I am using the bart large model.
summarization_pipeline = pipeline('summarization', model='facebook/bart-large-cnn')

# Load pre-trained classifier model. I took a bert-base-multilingual-uncased model 
# and specifically trained it on a dataset of 50 receipts and invoices each. It can 
# now distinguish between the two.
tokenizer = BertTokenizer.from_pretrained("fine-tuned-bert-classification-model")
classifier = BertForSequenceClassification.from_pretrained("fine-tuned-bert-classification-model")

# def classify_text(text):
def classify_text(text):
    encoding = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = classifier(**encoding)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return 'Document Type: Receipt' if prediction == 0 else 'Document Type: Invoice'

def summarize_text(text):
    summary = summarization_pipeline(text, max_length=300, min_length=150, do_sample=False)
    
    for summation in summary:
        print(summation['summary_text'])

# Load pre-trained model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def translate_text(text, src_lang="de", tgt_lang="en"):
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(translation[0])

def llm_pipeline(input = ""):
    # Example usage
    with open(input, "r") as f:
        text = f.read()

    summarize_text(text)
    print("\n")
    classify_text(text)
    translate_text(text)
