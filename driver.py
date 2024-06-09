from fetch_clean import prep
from llm_interface import llm_pipeline

file_path = input("Enter path to the file:")

prep_inp = prep(input=file_path)

llm_pipeline(input=prep_inp)
