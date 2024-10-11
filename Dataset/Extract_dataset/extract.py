import os
import json
import random
from tqdm import tqdm  # Import tqdm for the progress bar
from concurrent.futures import ThreadPoolExecutor, as_completed  # Para paralelización

# Function to read a file, attempting to decode it with UTF-8
def read_file_and_delete(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as file:
            content = file.readlines()

        # Filtrar las líneas que no empiezan con --- o +++
        content_without_changes = [
            line for line in content
            if not (line.startswith('---') or line.startswith('+++'))
        ]

        # Unir las líneas restantes en un solo texto
        filtered_content = ''.join(content_without_changes)
        
        return filtered_content

    except UnicodeDecodeError:
        print(f"Failed to read {path} with UTF-8. Trying another encoding...")

# Function to process a single folder and collect file data
def process_folder(root_folder, correct):
    folder_data = []  # Data for the folder
    
    files = os.listdir(root_folder)  # Get all files in the current folder
    content = []

    # Loop through the files in the folder and collect the necessary data
    for file in files:
        full_path = os.path.join(root_folder, file)  # Create full path for the file
        r_file = read_file_and_delete(full_path)
        if r_file:  # If file content exists, append it
            content.append(r_file)

    # Ensure that content is not empty before proceeding
    if content:
        for file_content in content:
            folder_data.append({
                'path': root_folder,
                'content': file_content,
                'correct': correct
            })

    return folder_data

# Function to load JSON data from the ASE dataset and save it (parallel version)
def load_json(source_directory, json_file, correct):
    data = []  # List to store the data from files

    # Count the total number of leaf folders
    leaf_folders = [root_folder for root_folder, subfolders, _ in os.walk(source_directory) if not subfolders]

    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as json_input:
                existing_data = json.load(json_input)
        except:
            existing_data = []
    else:
        existing_data = []

    # Parallel processing of leaf folders using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_folder, root_folder, correct): root_folder for root_folder in leaf_folders}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing leaf folders"):
            folder_data = future.result()
            if folder_data:
                data.extend(folder_data)

    # Save the updated data to the JSON file
    with open(json_file, 'w', encoding='utf-8') as json_output:
        json.dump(existing_data + data, json_output, indent=3, ensure_ascii=False)

# Function to count how many entries in the JSON file have the 'correct' field set to a specific value
def count_correct(json_path, correct):
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return sum(1 for element in data if element.get('correct') == correct)

def shuffle_json(file_path):
    with open(file_path, 'r') as archivo:
        datos = json.load(archivo)
    
    if isinstance(datos, list):
        random.shuffle(datos)
    elif isinstance(datos, dict):
        items = list(datos.items())
        random.shuffle(items)
        datos = dict(items)
    else:
        print("El formato del archivo no es ni lista ni diccionario. No se puede desordenar.")
        return

    with open(file_path, 'w') as archivo:
        json.dump(datos, archivo, indent=4)
    
    print(f"Archivo desordenado y guardado como '{file_path}'.")

# Paths to the dataset
largeC_path = "/home/hpc01/Marcos/Patch_Assesment/Dataset/Cache/patches/Large/correct"
largeO_path = "/home/hpc01/Marcos/Patch_Assesment/Dataset/Cache/patches/Large/overfitting"
smallC_path = "/home/hpc01/Marcos/Patch_Assesment/Dataset/Cache/patches/Small/correct"
smallO_path = "/home/hpc01/Marcos/Patch_Assesment/Dataset/Cache/patches/Small/overfitting"

large_json = "/home/hpc01/Marcos/Patch_Assesment/Dataset/json/large.json"
small_json = "/home/hpc01/Marcos/Patch_Assesment/Dataset/json/small.json"

# Process large dataset
load_json(largeC_path, large_json, True)
load_json(largeO_path, large_json, False)
print(f"ASE: \n\tCorrect: {count_correct(large_json, True)}\n\tIncorrect: {count_correct(large_json, False)}")

# Process small dataset
load_json(smallC_path, small_json, True)
load_json(smallO_path, small_json, False)
print(f"ASE: \n\tCorrect: {count_correct(small_json, True)}\n\tIncorrect: {count_correct(small_json, False)}")
