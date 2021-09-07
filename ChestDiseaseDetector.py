# %% markdown
## DISEASE DETECTOR FAST R-CNN
# %% codecell

# Para activar el directorio como path para cargar como módulos
# cuando se obtiene: 'No module named X'
import sys
sys.path.append('D:/GitHub/Mariuki/DiseaseDetector/')
# Activar variable de entorno por si se tiene el problema de que se reinicia el kernel
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %% codecell
# Hacer anotaciones sobre un conjunto de imagenes (Crear un dataset Artificial)
import pathlib
from visual import Annotator
from utils import get_filenames_of_path
directory = pathlib.Path('../data/Prueba') # Directorio de las imagenes a tomar
image_files = get_filenames_of_path(directory / 'Imagenes') # Cargar imágenes

annotator = Annotator(image_ids=image_files) # Cargar imágenes al visor
annotator.napari() # Abrir el visor (se abre en una ventana externa)
# En el visor de napari apareceran todas las imágenes del directorio una por una, y se puede
# Navegar entre ellas previonando las teblas 'n' y 'b'

## Etiquetar imágenes

# Con el visor Napari abierto en el conjunto de imagenes se crearan tantas capas sobre la imagen como
# clases de etiquetas se pongan, para llevar a acabo el proceso de etiquetado se selccionan las capas
# una a una y en el menú de la derecha se selecciona la figura de un cuadrado con vertices marcados
# para poder solocar la caja delimitadora correpsondiente en el lugar donde se ubica el objeto de interés
# de dicha clase, los cambios en las capas se guardan automaticamente.
# Por cada clase en cada imagen se deb correr la celda para crear la clase.

annotator.add_class(label='Clase 1', color='red') # Crear una clase - Label: nombre de etiqueta, color: color de las cajas delimitadoras
annotator.add_class(label='Clase 2', color='blue') # Crear una segunda clase

# Guardar las etiquetas de la imagen mostrada en el visor
save_dir = pathlib.Path(os.getcwd()) / 'Etiquetas' # Directorio para guardar la etiqueta
save_dir.mkdir(parents=True, exist_ok=True) # Verificar existencia
annotator.export(save_dir) # Guardar etiqueta como diccionario en archivo JSON

# Guardar todas las etiquetas disponibles en conjunto (de las imagenes que fueron etiquetadas)
save_dir = pathlib.Path(os.getcwd()) / '../Data/Prueba/Etiquetas' # Directorio para guardar las etiquetas de todas la imagenes al mismo tiempo
save_dir.mkdir(parents=True, exist_ok=True) # Verificar existencia del directorio
annotator.export_all(pathlib.Path(save_dir)) # Guardar todas las etiquetas


# %% codecell
## Mirar las imagenes con sus anotaciones
import pathlib
import torch
from utils import get_filenames_of_path
import json

root = pathlib.Path("data")
targets = get_filenames_of_path(root / 'ChestXRay8/ChestBBLabels')
targets.sort()

print(targets[0])
with open(targets[0]) as f:
    data = f.read()
annotation = json.loads(data)
# annotation["boxes"] = torch.tensor(annotation["boxes"]).to(torch.float32)

print(f'keys:\n{annotation.keys()}\n')
print(f'labels:\n{annotation["labels"]}\n')
print(f'boxes:\n{annotation["boxes"]}\n')
print(type(annotation["boxes"]))

# %% codecell
## Construcción de los conjuntos de datos
# Información y etiquetas estan contenidas en los directorios y son cargados al sistema
# Si las etiquetas son strings, las etiquetas deben ser codificadas como enteros.

# %% codecell
# Clase ObjectDetectionDataSet
# Se una la clase para crear el conjunto de datos
import pathlib
import albumentations as A
import numpy as np

from datasets import ObjectDetectionDataSet
from transformations import ComposeDouble, Clip, AlbumentationWrapper, FunctionWrapperDouble
from transformations import normalize_01
from utils import get_filenames_of_path, read_json, read_pt

# directorio
root = pathlib.Path("data/ChestXRay8")

# Entradas (imágenes) y Objetivos (etiquetas)
inputs = get_filenames_of_path(root / 'ChestBBImages')
targets = get_filenames_of_path(root / 'ChestBBLabels')

inputs.sort()
targets.sort()

# Mapeo de etiquetas a enteros
mapping = read_json(pathlib.Path('LabelsMappping.json'))
mapping
# Transformaciones y aumentado de datos
transforms = ComposeDouble([
    Clip(),
    # AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
    # AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
    # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# Conjunto de datos
dataset = ObjectDetectionDataSet(inputs=inputs,
                                 targets=targets,
                                 transform=transforms,
                                 add_dim = True,
                                 use_cache=False,
                                 convert_to_format=None,
                                 mapping=mapping)

## Mirando una muestra del conjunto de datos
sample = dataset[1]
# OLa muestra es un diccionario con las llaves:  ‘x’(Image), ‘x_name’(Image file name), ‘y’(Boxes), ‘y_name’(Annotations file name)
print(sample['x'].shape)
print(sample['x'])
## Visualiziar el conjunto de datos
colors = ['red','blue','black','purple','yellow','green','#aaffff','orange']
color_mapping = {v:colors[i] for i,v in enumerate(mapping.values())}

from visual import DatasetViewer

datasetviewer = DatasetViewer(dataset, color_mapping)
datasetviewer.napari()

# Si se requiere cofigurar algunas propiedades de visualización del conjunto de datos en napari
# es posible hacerlo abriendo una aplicación GUI, Esto debe correrse mientras el visualizador esta abierto.
datasetviewer.gui_text_properties(datasetviewer.shape_layer)

# %% codecell
## Fast R-CNN in PyTorch
# PyTorch tiene el módulo torchvision.models.detection.faster_rcnn.FasterRCNN el cuál
# tiene las siguientes especificaciones: La entrada del modelo debe ser una lista de tensores,
                              # cada una de la forma [C, H, W], una por cada imágen, y debe ser además
                              # en el rango 0-1. Diferentes imágenes pueden tener diferentes tamaños.

from torch.utils.data import DataLoader
from utils import collate_double

# Si la salida del objeto de la clase del conjunto de datos vienen en una forma diferente
# a la especificada por el módulo de Torch
# Es posible usar la función 'collate_double' para instanciar al cargador de datos.

dataloader = DataLoader(dataset=dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=0,
                        collate_fn=collate_double)

# Prueba de obtener un lote del conjunto de datos creado
batch = next(iter(dataloader))
print(batch)
# Dentro de el módulo de PyTorch Faster R-CNN Pse llevan acabo Transformaciones adicionales, dichas
# transformaciones pueden ser consultadas en torchvision.models.detection.transform.GeneralizedRCNNTransform
# Parámetros:
# * min_size (int): mínimo tamaño de la imagen a ser rescalada antes de ser procesada por la red.
# * max_size (int): máximo tamaño de la imagen a ser rescalada antes de ser procesada por la red.
# * image_mean (Tuple[float, float, float]): valores de media media utilizados para la normalización de entrada.
#   Estos son generalmente los valores medios del conjunto de datos en el cual fue entrenada la arquitectura base.
# * image_std (Tuple[float, float, float]): valores de  desviación estándar utilizados para la normalización de entrada.
#   Estos son generalmente los valores medios del conjunto de datos en el cual fue entrenada la arquitectura base.

## Verificar visualmente los datos despues de aplicar las tranformaciones del módulo - PyTorch GeneralizedRCNNTransform

colors = ['red','blue','black','purple','yellow','green','#aaffff','orange']
color_mapping = {v:colors[i] for i,v in enumerate(mapping.values())}

from visual import DatasetViewer
from torchvision.models.detection.transform import GeneralizedRCNNTransform

transform = GeneralizedRCNNTransform(min_size=1024,
                                     max_size=1024,
                                     image_mean=[0.485, 0.456, 0.406],
                                     image_std=[0.229, 0.224, 0.225])

datasetviewer = DatasetViewer(dataset, color_mapping, rccn_transform=transform)
datasetviewer.napari()

# Para observar el impacto de la transformación se pueden verificar algunas estadísticas
# del conjunto de datos con o sin estas transformaciones:

from utils import stats_dataset

stats = stats_dataset(dataset)

from torchvision.models.detection.transform import GeneralizedRCNNTransform

transform = GeneralizedRCNNTransform(min_size=1024,
                                     max_size=1024,
                                     image_mean=[0.485, 0.456, 0.406],
                                     image_std=[0.229, 0.224, 0.225])

stats_transform = stats_dataset(dataset, transform)

import pandas as pd
stats_comparison = pd.DataFrame({'Image Height':[stats['image_height'].max(),stats_transform['image_height'].max()],
                                 'Image Width':[stats['image_width'].max(),stats_transform['image_width'].max()],
                                 'Image Mean':[stats['image_mean'].max(),stats_transform['image_mean'].max()],
                                 'Image Std':[stats['image_std'].max(),stats_transform['image_std'].max()],
                                 'Boxes Height':[stats['boxes_height'].max(),stats_transform['boxes_height'].max()],
                                 'Boxes Width':[stats['boxes_width'].max(),stats_transform['boxes_width'].max()],
                                 'Boxes Num':[stats['boxes_num'].max(),stats_transform['boxes_num'].max()],
                                 'Boxes Area':[stats['boxes_area'].max(),stats_transform['boxes_area'].max()]}, index = ['Without Trsfms', 'With Trsfms'])

print(stats_comparison.T)

# %% codecell
## Cajas Ancla

from torchvision.models.detection.transform import GeneralizedRCNNTransform
from pytorch_faster_rcnn_tutorial.visual import AnchorViewer

transform = GeneralizedRCNNTransform(min_size=1024,
                                     max_size=1024,
                                     image_mean=[0.485, 0.456, 0.406],
                                     image_std=[0.229, 0.224, 0.225])

image = dataset[0]['x']  # ObjectDetectionDataSet
print(image.shape)
feature_map_size = (512, 32, 32)
anchorviewer = AnchorViewer(image=image,
                            rcnn_transform=transform,
                            feature_map_size=feature_map_size,
                            anchor_size=((128, 256, 512),),
                            aspect_ratios=((1.0,),)
                            )
anchorviewer.napari()

# Identificando el valor feature_map_size mediante el envío de un dummy torch.tensor a través de la arquitectura base (pre-entrenada):
# from torchsummary import summary
# summary = summary(backbone, (3, 1024, 1024))

# %% Code cell

## Entrenamiento del modelo
# Usando una API de alto nivel para obtener funcionalidades integradas y características como logging, metrics,
# early stopping, mixed precision training, etc
import pathlib

import albumentations as A
import numpy as np
from torch.utils.data import DataLoader

from datasets import ObjectDetectionDataSet
from transformations import ComposeDouble, Clip, AlbumentationWrapper, FunctionWrapperDouble
from transformations import normalize_01
from utils import get_filenames_of_path, collate_double, read_json

# Hiper-parámetros
params = {'BATCH_SIZE': 4,
          'LR': 0.001,
          'PRECISION': 32,
          'CLASSES': 8,
          'SEED': 42,
          'PROJECT': 'Chests',
          'EXPERIMENT': 'chests',
          'MAXEPOCHS': 500,
          'BACKBONE': 'resnet18',
          'FPN': False, # Activar uso o no de FPN
          'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
          'MIN_SIZE': 1024,
          'MAX_SIZE': 1024,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }

# directorio Raíz
root = pathlib.Path("data/ChestXRay8")

# Cargar entradas y objetivos
inputs = get_filenames_of_path(root / 'ChestBBImages')
targets = get_filenames_of_path(root / 'ChestBBLabels')

inputs.sort()
targets.sort()

# Mapeo de clases a un valor entero
mapping = read_json(pathlib.Path('LabelsMappping.json'))
mapping

# Transformaciones de entranmeinto y aumentado de datos
transforms_training = ComposeDouble([
    Clip(),
    AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
    AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
    # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# Transformaciones de validación
transforms_validation = ComposeDouble([
    Clip(),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# Transformaciones de prueba
transforms_test = ComposeDouble([
    Clip(),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# Semilla de aleatoreidad
from pytorch_lightning import seed_everything

seed_everything(params['SEED'])
# len(inputs)
# Division del conjunto de datos en subconjuntos (entrenamiento, validación y prueba)
inputs_train, inputs_valid, inputs_test = inputs[:int(len(inputs)*0.7)], inputs[int(len(inputs)*0.7):int(len(inputs)*0.8)], inputs[int(len(inputs)*0.8):]
targets_train, targets_valid, targets_test = targets[:int(len(inputs)*0.7)], targets[int(len(inputs)*0.7):int(len(inputs)*0.8)], targets[int(len(inputs)*0.8):]

# Conjunto de datos de entrenamiento
dataset_train = ObjectDetectionDataSet(inputs=inputs_train,
                                       targets=targets_train,
                                       transform=transforms_training,
                                       add_dim = True,
                                       use_cache=True,
                                       convert_to_format=None,
                                       mapping=mapping,
                                       tgt_int64=True)

# Conjunto de datos de  validación
dataset_valid = ObjectDetectionDataSet(inputs=inputs_valid,
                                       targets=targets_valid,
                                       transform=transforms_validation,
                                       add_dim = True,
                                       use_cache=True,
                                       convert_to_format=None,
                                       mapping=mapping,
                                       tgt_int64=True)

# Conjunto de datos de prueba
dataset_test = ObjectDetectionDataSet(inputs=inputs_test,
                                      targets=targets_test,
                                      transform=transforms_test,
                                      add_dim = True,
                                      use_cache=True,
                                      convert_to_format=None,
                                      mapping=mapping,
                                      tgt_int64=True)

# Cargador de datos de entrenamiento
dataloader_train = DataLoader(dataset=dataset_train,
                              batch_size=params['BATCH_SIZE'],
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_double)

# Cargador de datos de validacion
dataloader_valid = DataLoader(dataset=dataset_valid,
                              batch_size=4,
                              shuffle=False,
                              num_workers=0,
                              collate_fn=collate_double)

# Cargador de datos de prueba
dataloader_test = DataLoader(dataset=dataset_test,
                             batch_size=2,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_double)

# %% codecell

## Neptune es un software para dar seguimiento al proceso de ejecución de entrenamiento y las evaluaciones de los modelos.
# Es similar a CSV logger, TensorBoard or MLflow

# Cargador de Neptune
from pytorch_lightning.loggers.neptune import NeptuneLogger
# from api_key_neptune import get_api_key
import os
import neptune
import neptune.new as neptune
# from getpass import getpass
# api_key = getpass('eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMjQ1NGZkNS00MmJhLTQwYWYtYjEyYi02ZTFjY2JkN2Q2YzMifQ==')
# print(api_key)
# api_key_neptune.py
#
# def get_api_key():
#     return 'your_super_long_API_token'

# %env NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'
# os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"]="TRUE"
# run = neptune.init(project='rubsini/DiseasesDetection',api_token=api_key) # your credentials
# api_key = get_api_key()
# print(run)
api_key = os.getenv("NEPTUNE") # cuando ya se tiene configurada la llave como variable de entorno

neptune_logger = NeptuneLogger(
    api_key=api_key,
    project_name=f'rubsini/DiseasesDetection',
    experiment_name=params['EXPERIMENT'],
    params=params
) # Conexión con neptune


## Pruebas de importación de modelos ResNet
# import torchvision.models as models
# import torch
# from urllib.request import urlopen

# model = models.resnet18(pretrained=True)

# Inicialización del modelo
from faster_RCNN import get_fasterRCNN_resnet

model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                              backbone_name=params['BACKBONE'],
                              anchor_size=params['ANCHOR_SIZE'],
                              aspect_ratios=params['ASPECT_RATIOS'],
                              fpn=params['FPN'],
                              min_size=params['MIN_SIZE'],
                              max_size=params['MAX_SIZE'])

# Inicialización de módulo Lightning
from faster_RCNN import FasterRCNN_lightning

task = FasterRCNN_lightning(model=model, lr=params['LR'], iou_threshold=params['IOU_THRESHOLD'])


# Crear el entrenador con monitoreos específicos
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

checkpoint_callback = ModelCheckpoint(monitor='Validation_mAP', mode='max')
learningrate_callback = LearningRateMonitor(logging_interval='step', log_momentum=False)
early_stopping_callback = EarlyStopping(monitor='Validation_mAP', patience=50, mode='max')

# Inicializar el entrenador
from pytorch_lightning import Trainer

trainer = Trainer(gpus=1,
                  precision=params['PRECISION'],  # Al probar con 16, enable_pl_optimizer=False
                  callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
                  default_root_dir="Experiments/",  # Directorio para guardar los checkpoints
                  logger=neptune_logger,
                  log_every_n_steps=1,
                  num_sanity_val_steps=0#,
                  # enable_pl_optimizer=False,  # Se descomenta cuando se usa precisión de 16
                  )
# %% Comenzar el entrenamiento
trainer.max_epochs = params['MAXEPOCHS']
trainer.fit(task,
            train_dataloader=dataloader_train,
            val_dataloaders=dataloader_valid)

# %% codecell
## Obtener el mejor modelo y usarlo para predecir la información de prueba, basado en el conjunto de datos de validación y
# conforme a la metrica usada (mAP from pascal VOC)
trainer.test(ckpt_path='best', test_dataloaders=dataloader_test)

# Cargar información adicional del esperimento, como: todos los paquetes y versiones de el ambiente conda que fue utilizado a neptune.
from utils import log_packages_neptune
import utils
from neptunecontrib.api import log_table
import importlib_metadata
def log_packages_neptune(neptune_logger):
    """Uses the neptunecontrib.api to log the packages of the current python env."""
    dists = importlib_metadata.distributions()
    packages = {
        idx: (dist.metadata["Name"], dist.version) for idx, dist in enumerate(dists)
    }

    packages_df = pd.DataFrame.from_dict(
        packages, orient="index", columns=["package", "version"]
    )

    log_table(name="packages", table=packages_df, experiment=neptune_logger.experiment)

log_packages_neptune(neptune_logger)

# Cargar el mapeo como una tabla a neptune
from utils import log_mapping_neptune

def log_mapping_neptune(mapping: dict, neptune_logger):
    """Uses the neptunecontrib.api to log a class mapping."""
    mapping_df = pd.DataFrame.from_dict(
        mapping, orient="index", columns=["class_value"]
    )
    log_table(name="mapping", table=mapping_df, experiment=neptune_logger.experiment)

log_mapping_neptune(mapping, neptune_logger)

# Cargar el modelo a neptune
from utils import log_model_neptune

checkpoint_path = pathlib.Path(checkpoint_callback.best_model_path)
log_model_neptune(checkpoint_path=checkpoint_path,
                  save_directory=pathlib.Path.home(),
                  name='best_model.pt',
                  neptune_logger=neptune_logger)

## Realizar mas experimentos en datos no vistos (inferencia)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import ast
import pathlib

import neptune
import numpy as np
import torch
from torch.utils.data import DataLoader

# from api_key_neptune import get_api_key
from pytorch_faster_rcnn_tutorial.datasets import ObjectDetectionDatasetSingle, ObjectDetectionDataSet
from pytorch_faster_rcnn_tutorial.transformations import ComposeSingle, FunctionWrapperSingle, normalize_01, ComposeDouble, FunctionWrapperDouble
from utils import get_filenames_of_path, collate_single

# Parámetros
params = {'EXPERIMENT': 'CAR-10',
          'INPUT_DIR': "../Experiments/tests", # Cargar los archivos a predecir
          'PREDICTIONS_PATH': '../Experiments/predictions', # Donde guardar las predicciones
          'MODEL_DIR': '../data/chests/DIS-10/checkpoints/epoch=86-step=521.ckpt', # Cargar el modelo el checkpoint
          'DOWNLOAD': False, # Si se debe descargar el modelo de Neptune
          'DOWNLOAD_PATH': 'Experimentos/', # Donde se guarda el modelo
          'OWNER': 'Username',
          'PROJECT': 'DiseasesDetection',
          }

# Archivos de entrada
inputs = get_filenames_of_path(pathlib.Path(params['INPUT_DIR']))
inputs.sort()

# Transformaciones
transforms = ComposeSingle([
    FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
    FunctionWrapperSingle(normalize_01)
])

# Crear el conjunto y el cargador de los datos
dataset = ObjectDetectionDatasetSingle(inputs=inputs,
                                       transform=transforms,
                                       use_cache=False,
                                       )

dataloader_prediction = DataLoader(dataset=dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=collate_single)

# Importar el experimento de neptune
api_key = os.getenv("NEPTUNE") # cuando ya se tiene configurada la llave como variable de entorno
project_name = f'{params["OWNER"]}/{params["PROJECT"]}'
project = neptune.init(project_qualified_name=project_name, api_token=api_key)  # Obtener el proyecto
experiment_id = params['EXPERIMENT']  # Identificador del experiemento
experiment = project.get_experiments(id=experiment_id)[0]
parameters = experiment.get_parameters()
properties = experiment.get_properties()

# Visualizar el conjunto de datos
from pytorch_faster_rcnn_tutorial.visual import DatasetViewerSingle
from torchvision.models.detection.transform import GeneralizedRCNNTransform

transform = GeneralizedRCNNTransform(min_size=int(parameters['MIN_SIZE']),
                                     max_size=int(parameters['MAX_SIZE']),
                                     image_mean=ast.literal_eval(parameters['IMG_MEAN']),
                                     image_std=ast.literal_eval(parameters['IMG_STD']))


datasetviewer = DatasetViewerSingle(dataset, rccn_transform=None)
# datasetviewer.napari()

# Descargar el modelo de Neptune.ai or o cargar del checkpoint
if params['DOWNLOAD']:
    download_path = pathlib.Path(params['DOWNLOAD_PATH'])
    model_name = properties['checkpoint_name'] # logged when called log_model_neptune()
    if not (download_path / model_name).is_file():
        experiment.download_artifact(path=model_name, destination_dir=download_path)  # download model

    model_state_dict = torch.load(download_path / model_name)
else:
    checkpoint = torch.load(params['MODEL_DIR'])
    model_state_dict = checkpoint['hyper_parameters']['model'].state_dict()

# Inicialización del modelo
from pytorch_faster_rcnn_tutorial.faster_RCNN import get_fasterRCNN_resnet
model = get_fasterRCNN_resnet(num_classes=int(parameters['CLASSES']),
                              backbone_name=parameters['BACKBONE'],
                              anchor_size=ast.literal_eval(parameters['ANCHOR_SIZE']),
                              aspect_ratios=ast.literal_eval(parameters['ASPECT_RATIOS']),
                              fpn=ast.literal_eval(parameters['FPN']),
                              min_size=int(parameters['MIN_SIZE']),
                              max_size=int(parameters['MAX_SIZE'])
                              )

# Cargar pesos
model.load_state_dict(model_state_dict)

## Iterar a través del conjunto de datos y predecir las cajas delimitadoras de cada imagen.
# Inferencia
model.eval()
for sample in dataloader_prediction:
    x, x_name = sample
    with torch.no_grad():
        pred = model(x)
        pred = {key: value.numpy() for key, value in pred[0].items()}
        name = pathlib.Path(x_name[0])
        torch.save(pred, pathlib.Path(params['PREDICTIONS_PATH']) / name.with_suffix('.pt'))

# Para visualizar los resultados se puede crear un conjunto de datos predichos
predictions = get_filenames_of_path(pathlib.Path(params['PREDICTIONS_PATH']))
predictions.sort()
# print(read_pt(predictions[0]))

transforms_prediction = ComposeDouble([
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

dataset_prediction = ObjectDetectionDataSet(inputs=inputs,
                                            targets=predictions,
                                            transform=transforms_prediction,
                                            use_cache=False)

# print('Objectdataset: ', dataset_prediction[2])
# Visualizar predicciones
from pytorch_faster_rcnn_tutorial.visual import DatasetViewer

color_mapping = {
    1: 'red',
}

datasetviewer_prediction = DatasetViewer(dataset_prediction, color_mapping)
datasetviewer_prediction.napari()


# %% codecell
import json
from pytorch_faster_rcnn_tutorial.utils import read_pt
import pytorch_faster_rcnn_tutorial.utils as ut

pr = get_filenames_of_path(pathlib.Path('../Experiments/Predicciones/'))
print(pr[0])
print(read_pt(pr[0])['scores'])
ml = torch.load(pr[0])
print(ml)

with open(pr[0], "r", encoding='cp850') as fp:
    print(fp)
    file = json.loads(s=fp.read())
