# For OCR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pdf2image import convert_from_path
# To check validity of URLs
from urllib.parse import urlparse
# For cleaning text 
import re
from bs4 import BeautifulSoup
# For normalising text
from nltk.corpus import stopwords
# For tokenization
import nltk
from transformers import BertTokenizer
# Because when isn't numpy needed 
import numpy as np

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def process_document(file_path = "input.txt", out_dir = "./"):
    # Initialise the ocr model
    model = ocr_predictor(pretrained=True, det_arch= "db_resnet50", reco_arch = "crnn_vgg16_bn", assume_straight_pages=False)
    
    # Get the file name
    file = file_path.split("/")[-1]
    
    # Get the file type
    file_type = file.split(".")[-1]

    # Get the file name by removing the file_type from file
    file_name = file.replace(file_type, "")

    # Open the doc
    if is_valid_url(file_path):
        file_path = [DocumentFile.from_url(file_path)]
    elif file_type == "pdf":
        doc = convert_from_path(file_path)
        doc = [np.array(img) for img in doc]
    elif file_type in ["png", "jpg", "jpeg"]:
        doc = [DocumentFile.from_images(file_path)]
    else:
        raise Exception("Invalid file type or URL")
    
    # Get the results
    result = model(doc)

    # Write the extracted text to a .txt file
    with open(out_dir + file_name + '.txt', 'w') as f:
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        f.write(word.value + " ")
                    f.write("\n")

    return out_dir + file_name + ".txt"

def clean_text(unclean_text = ""):
    # Get rid of all HTML tags, if any.
    text = BeautifulSoup(unclean_text, "html.parser").get_text()
    
    # Now, we will use regex to get rid of unwanted symbols/punctuation.
    # We shall preserve addresses to websites and email addresses as well.

    url_pattern = r'\b(?:http[s]?://|www\.)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:[A-Z|a-z]{2,})\b'
    urls = re.findall(url_pattern, text)

    for i, url in enumerate(urls):
        text = text.replace(url, f'URL{i}')

    # Remove unwanted characters
    text = re.sub(r'[^A-Za-z0-9\s\.]+', '', text)

    # Replace the placeholders with the original email addresses and URLs
    for i, url in enumerate(urls):
        text = text.replace(f'URL{i}', url)

    return text

def normalise_text(unnormalised_text = ""):
    # Fetch and collect the stopwords in english so that we may remove them
    stop_words = set(stopwords.words('english'))

    # We will lowercase all the text in the doc, to avoid misrepresenting 
    # email-ids and urls we will remove them from the text and add them back
    # once the text has been reduced to lowercase.
    url_pattern = r'\b(?:http[s]?://|www\.)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:[A-Z|a-z]{2,})\b'
    urls = re.findall(url_pattern, unnormalised_text)

    for i, url in enumerate(urls):
        text = unnormalised_text.replace(url, f'URL{i}')
    
    text = text.lower()
    lines = text.split("\n")
    new_text = ""
    for line in lines:
        new_line = ""
        words = line.split(" ")
        for word in words:
            if word not in stop_words:
                new_line = ' '.join([new_line, word])
        new_text = "\n".join([new_text, new_line])

    for i, url in enumerate(urls):
        new_text = new_text.replace(f'url{i}', url)
    
    return new_text

def tokenize_text(text = ""):
    # 
    text = nltk.sent_tokenize(text)
    text = " ".join(text)

    return text

def preprocess_text(file = ""):
    # Read the text file
    with open(file, 'r') as f:
        # Fetch and store the text
        text = f.read()

        # Clean the text
        cleaned_text = clean_text(text)

        # Normalise the text
        normalised_text = normalise_text(cleaned_text)

        # Tokenize the text
        tokenized_text = tokenize_text(normalised_text)

    return tokenized_text

# store the name of the text file with the data.

def prep(input = "", out_dir="./"):
    out = process_document(file_path=input, out_dir=out_dir)

    # preprocess the text, cleaning and tokenizing it to prepare for llm inference.
    tokens = preprocess_text(out)

    with open(out, "w") as f:
        f.write(tokens)

    # Return the name of the file containing the processed text
    return out
