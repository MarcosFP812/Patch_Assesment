from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint = "meta-llama/Llama-3.1-8B"
model_path = "/home/hpc01/Marcos/Patch_Assesment/Model"
tokenizer_path = "/home/hpc01/Marcos/Patch_Assesment/Tokenizer"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
new_tokens = {"additional_special_tokens": ["<|pad|>"]}   
tokenizer.add_special_tokens(new_tokens)

model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)







