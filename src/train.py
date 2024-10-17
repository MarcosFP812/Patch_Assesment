from transformers import AutoTokenizer, AutoModelForSequenceClassification, DefaultDataCollator
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from Model import PatchALlama
#from accelerate import Accelerator
import tqdm
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

model_path = "meta-llama/Llama-3.2-1B"
tokenizer_path = "/home/hpc01/Marcos/Patch_Assesment/Tokenizer"
tokenized_dataset_path = "/home/hpc01/Marcos/Patch_Assesment/Dataset/TokenizedDatasets/large"

#model = PatchALlama(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenized_datasets = load_from_disk(tokenized_dataset_path)

pad_token_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")  # Obtener el id de <|pad|>
batch_size = 4

def create_batches(batch):
        """
        Iterador que divide el dataset en batches y añade padding para igualar las secuencias al tamaño máximo del batch.

        Args:
        - dataset: dataset tokenizado (e.g., tokenized_datasets["train"]).
        - batch_size: tamaño del batch.
        - pad_token_id: id del token de padding (en este caso, el id de <|pad|>).

        Yields:
        - batch: un batch que contiene 'input_ids', 'attention_mask' y 'labels'.
        """
        # Iterar sobre el dataset en pasos del tamaño del batch
        

        batch_input_ids = [aux['input_ids'] for aux in batch]
        for i in range(len(batch_input_ids)):
            print(len(batch_input_ids[i]))
        batch_attention_mask = [aux['attention_mask'] for aux in batch]
        batch_labels = [aux['labels'] for aux in batch]
        

        # Encontrar la longitud máxima de 'input_ids' en el batch actual
        max_length = max(len(input_ids) for input_ids in batch_input_ids)

        # Crear listas para almacenar los input_ids y attention_mask con padding
        padded_input_ids = []
        padded_attention_mask = []

        # Aplicar padding a cada secuencia del batch
        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):

            # Calcular cuántos tokens de padding se necesitan
            padding_length = max_length - len(input_ids) + 1
            
            # Rellenar con el token de padding
            padded_input_ids.append(input_ids + [128001] + [pad_token_id] * (padding_length-1))
            padded_attention_mask.append(attention_mask + [0] * padding_length)  # 0 para los tokens de padding
        
        # Yield del batch actual
        r =  {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long)
        }
        for i in range(len(batch_input_ids)):
            print(r[i])
        return r



# Inicializar el iterador
#batch_gen_train = create_batches(tokenized_datasets["train"], batch_size, pad_token_id)
#batch_gen_test = create_batches(tokenized_datasets["test"], batch_size, pad_token_id)

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=create_batches
)

for batch in train_dataloader:
    break

print({k: v.shape for k, v in batch.items()})



"""
accelerator = Accelerator()

optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    batch_gen_train, batch_gen_test, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

model.save_pretrained("/home/hpc01/Marcos/Patch_Assesment/PatchALlama")

"""