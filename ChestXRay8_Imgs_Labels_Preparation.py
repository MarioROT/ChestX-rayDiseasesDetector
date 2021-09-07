# %% markdown
## Extraer las imagenes que tienen etiquetadod e caja delimittadora por padecimeinto
### Crear un archivo JSON por cada imagen con las respectivas etiquetas de padecimeintos identificados

# %% codecell
import pandas as pd
import numpy as np
import pathlib
from utils import read_json
import json
from shutil import copyfile

# %% codecell
# Cargar el archivo de los datos de imagenes con etiquetado de cajas delimitadoras
BBAnot = pd.read_csv('../ChestX-ray8-Data/BBox_List_2017.csv') # Leer archivo
BBAnot = BBAnot[['Image Index','Finding Label', 'Bbox [x', 'y','w','h]']] # Seleccionar columnas de interés
BBAnot = BBAnot.sort_values(by=['Image Index']).reset_index().drop(['index'], axis=1) # Ordenar por índice
BBAnotRep = BBAnot[BBAnot.duplicated(['Image Index'], keep=False)] # Verificar las imagenes que tienen mas de una anotación
BBImgs = BBAnot['Image Index'].unique() # Obtener los nombres de las imágenes que tienen etiquetado de caja delimitadora
labels = BBAnot['Finding Label'].unique() # Obtener las clases (nombres de los padecimeintos)
numlabs = {key:r  for r,key in enumerate(labels)} # create a labels mapping dictionary ('string':int)
# filelabs = open('LabelsMappping.json', "x") # Save labels mapping
# json.dump(numlabs, filelabs)
# filelabs.close()

# Poner los datos de cajas delimitadoras en una sola lista y extraer la correspondiente etiqueta
BBAnot['boxes'] = [[BBAnot.iloc[i].loc['Bbox [x'],BBAnot.iloc[i].loc['y'],BBAnot.iloc[i].loc['Bbox [x']+BBAnot.iloc[i].loc['w'],BBAnot.iloc[i].loc['y']+BBAnot.iloc[i].loc['h]']] for i in range(len(BBAnot))]
boxes = [[int(BBAnot.iloc[i].loc['Bbox [x']),int(BBAnot.iloc[i].loc['y']),int(BBAnot.iloc[i].loc['Bbox [x']+BBAnot.iloc[i].loc['w']),int(BBAnot.iloc[i].loc['y']+BBAnot.iloc[i].loc['h]'])] for i in range(len(BBAnot))]
labs = [str(BBAnot.iloc[i].loc['Finding Label']) for i in range(len(BBAnot))]
BBAnotRep

# %% codecell
## Extraer la información de la caja delimitadora por imagen y guardarla en el archivo JSON correspondiente a la imagen
# Si existe mas de una anotación por imagen es guardada en el archivo creado anteriormente para la correspondiente imagen
created = []
for i in range(len(BBAnot)):
    root = pathlib.Path('data/ChestXRay8/ChestBBLabels/' + BBAnot.iloc[i].loc['Image Index'][:-4] + ".json")
    if BBAnot.iloc[i].loc['Image Index'] in created: # Checar si ya existe un archivo para dicah imagen
        lab = read_json(root)
        lab['labels'].append(BBAnot.iloc[i].loc['Finding Label'])
        lab['boxes'].append(BBAnot.iloc[i].loc['boxes'])
        r = {'labels':[str(k) for k in lab['labels']],'boxes':[[int(j) for j in k] for k in lab['boxes']]}
        file = open(root, "w")
        json.dump(r, file)
        file.close()
    else: # Crear un archivo para dicha imagen
        file = open(root, "x")
        lab = {'labels':[str(BBAnot.iloc[i].loc['Finding Label'])],
               'boxes':[[int(j) for j in BBAnot.iloc[i].loc['boxes']]]}
        json.dump(lab, file)
        file.close()
        created.append(BBAnot.iloc[i].loc['Image Index'])

# %% codecell
# Hacer una copia en otro directorio unicamente de las 880 imágenes que tienen
# etiquetado de caja delimitadora del total de imagenes (112,220)

inputs = BBImgs
inputs = [pathlib.Path('images/images/' + i ) for i in inputs]
inputs.sort()
len(inputs)

for idx, path in enumerate(inputs):
    copyfile(path, 'data/ChestXRay8/ChestBBImages/' + path.stem + path.suffix)
