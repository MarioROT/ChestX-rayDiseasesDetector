import sys
sys.path.append('D:/GitHub/Mariuki/DiseaseDetector/Detector de Padecimientos Rayos-X Torax - Codigo')
sys.path
# import _openssl error: (DLLS)
# https://stackoverflow.com/a/60405693/12283874
# https://stackoverflow.com/a/64054522/12283874

# Activar variable de entorno por si se tiene el problema de que se reinicia el kernel
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pathlib
import albumentations as A
import numpy as np
import operator as op

from datasets import ObjectDetectionDataSet
from transformations import ComposeDouble, Clip, AlbumentationWrapper, FunctionWrapperDouble
from transformations import normalize_01, addHM
from utils import get_filenames_of_path, read_json, read_pt

# directorio
root = pathlib.Path("data/ChestXRay8")

# Entradas (imágenes) y Objetivos (etiquetas)
inputs = get_filenames_of_path(root / 'ChestBBImages')
targets = get_filenames_of_path(root / 'ChestBBLabels')

# # directorio
# root = pathlib.Path("data/Prueba")
#
# # Entradas (imágenes) y Objetivos (etiquetas)
# inputs = get_filenames_of_path(root / 'Imagenes')
# targets = get_filenames_of_path(root / 'Etiquetas')

inputs.sort()
targets.sort()

# Mapeo de etiquetas a enteros
mapping = read_json(pathlib.Path('Detector de Padecimientos Rayos-X Torax - Codigo/LabelsMappping.json'))
mapping
# mapping = {'Clase 1':0, 'Clase 2': 1}
# Transformaciones y aumentado de datos
transforms = ComposeDouble([
    Clip(),
    # AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
    # AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
    # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
    # FunctionWrapperDouble(np.moveaxis, source=-1, destination=0), # Solo aplica cuando las imagenes son originalmente de 3 canales de color
    FunctionWrapperDouble(normalize_01),
    # FunctionWrapperDouble(addHM)
])

# Conjunto de datos
dataset = ObjectDetectionDataSet(inputs=inputs,
                             targets=targets,
                             transform=transforms,
                             add_dim = True,
                             use_cache=False,
                             convert_to_format=None,
                             mapping=mapping)#,
                             # metadata_dir='ChestX-ray8-Data/Data_Entry_2017_v2020.csv',
                             # filters = [[op.gt,'Patient Age',10],[op.lt,'Patient Age',81]],
                             # id_column = 'Image Index')

# Adquiriendo la cantidad de datos del conjunto (si es el caso, con los filtros aplicados)
len(dataset)

## Mirando una muestra del conjunto de datos
sample = dataset[1]

# La muestra es un diccionario con las llaves:  ‘x’(Image), ‘x_name’(Image file name), ‘y’(Boxes), ‘y_name’(Annotations file name)
print(sample['x'].shape)
print(sample['x'])

## Visualiziar el conjunto de datos
colors = ['red','blue','black','purple','yellow','green','#aaffff','orange']
color_mapping = {v:colors[i] for i,v in enumerate(mapping.values())}

from visual import DatasetViewer

datasetviewer = DatasetViewer(dataset, color_mapping)
datasetviewer.napari() # como son primero canal por eso salen en 3 capas en grises incluso las imagenes de color, si tienen los canales como ultima dimensión ya salen a color

# Si se requiere cofigurar algunas propiedades de visualización del conjunto de datos en napari
# es posible hacerlo abriendo una aplicación GUI, Esto debe correrse mientras el visualizador esta abierto.
datasetviewer.gui_text_properties(datasetviewer.shape_layer)
