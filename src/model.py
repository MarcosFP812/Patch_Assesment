from transformers import AutoModelForSequenceClassification, AutoTokenizer

model="meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model)






