# Patch_Assesment
 
## Roadmap

- Obtener dataset de CACHE: Large y Small, borrando los comentarios y comprobando que no haya Overlap (Por ahora no lo tenemos en cuenta) 

Done y con multithreading (Mu rapido), están en large.json y small.json

- Cargar el modelo y cambiar la cabeza (en principio con AutoModelForClasification)

- Tratar los datos para tokenizerlos y con dataloader para dividir en batches

- Crear el training loop con accelerate (tener en cuenta en el futuro el gradient accuulator y el gradient_checkpoint)



## Resultados

- Resultado ejecución de extract.py
```python
Processing leaf folders: 100%|█| 17960/17960 [00:00<00:00, 52505.72it/
Processing leaf folders: 100%|█████| 660/660 [00:06<00:00, 108.43it/s]
Archivo desordenado y guardado como '/home/hpc01/Marcos/Patch_Assesment/Dataset/json/large.json'.
ASE:
        Correct: 25589
        Incorrect: 24105
Processing leaf folders: 100%|███████| 54/54 [00:00<00:00, 691.18it/s]
Processing leaf folders: 100%|██████| 75/75 [00:00<00:00, 1655.02it/s]
Archivo desordenado y guardado como '/home/hpc01/Marcos/Patch_Assesment/Dataset/json/small.json'.
ASE:
        Correct: 535
        Incorrect: 648
```



