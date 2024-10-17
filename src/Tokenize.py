from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer_path = "/home/hpc01/Marcos/Patch_Assesment/Tokenizer"
dataset_path = "/home/hpc01/Marcos/Patch_Assesment/Dataset/json/small.json"
save_path = "/home/hpc01/Marcos/Patch_Assesment/Dataset/TokenizedDatasets/small"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def tokenize_function(example):
    return tokenizer(example["content"], truncation = True)

def tokenize_dataset_large(dataset_path, save_path):
    raw_dataset = load_dataset('json', data_files = dataset_path) 
    raw_dataset_split = raw_dataset['train'].train_test_split(test_size=0.2)

    tokenized_datasets = raw_dataset_split.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["path", "content"])
    tokenized_datasets = tokenized_datasets.rename_column("correct", "labels")

    tokenized_datasets.save_to_disk(save_path)

def tokenize_dataset_small(dataset_path, save_path):
    raw_dataset = load_dataset('json', data_files = dataset_path) 
    raw_dataset_split = raw_dataset['train'].train_test_split(test_size=1)

    tokenized_datasets = raw_dataset_split.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["path", "content"])
    tokenized_datasets = tokenized_datasets.rename_column("correct", "labels")

    tokenized_datasets.save_to_disk(save_path)

tokenize_dataset_small(dataset_path, save_path)



