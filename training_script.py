import sys
sys.path.append('D:/GitHub/Mariuki/DiseaseDetector/Detector de Padecimientos Rayos-X Torax - Codigo')
sys.path.append('D:/GitHub/Mariuki/DiseaseDetector')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pathlib
import albumentations as A
import numpy as np
from torch.utils.data import DataLoader
import torch

from datasets import ObjectDetectionDataSet
from transformations import ComposeDouble, Clip, AlbumentationWrapper, FunctionWrapperDouble, normalize_01,RescaleWithBB
from utils import get_filenames_of_path, collate_double, read_json, log_mapping_neptune,log_model_neptune, log_packages_neptune
from neptunecontrib.api import log_table
import importlib_metadata

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor,EarlyStopping

from faster_RCNN import FasterRCNN_lightning
from faster_RCNN import get_fasterRCNN_mobilenet, get_fasterRCNN_resnet, get_fasterRCNN_mobilenet, get_fasterRCNN_shufflenet_v2, get_fasterRCNN_efficientnet


# Hiperparámetros
params = {'OWNER': 'rubsini',  # Nombre de usuario en Neptune.ai
          'SAVE_DIR': "../Experiments/",  # Directorio para guardar los checkpoints en entrenamiento
          'PROJECT': 'DiseasesDetection', # Nombre dle proyecto creado en Neptune.ai
          'EXPERIMENT': 'chests', # nombre del experimento dentro del proyecto
          'LOG_MODEL': True,  # Si se cargará el modelo a neptune después del entrenamiento
          'GPU': 1,  # Activar o  desactivar para elegir entrenar en GPU o CPU
          'BATCH_SIZE': 24, # Tamaño del lote
          'LR': 0.01, # Tasa de aprendizaje
          'PRECISION': 16, # Precisión de cálculo
          'CLASSES': 9, # Número de clases (incluyendo el fondo)
          'SEED': 42, # Semilla de aleatoreidad
          'MAXEPOCHS': 500, # Número máximo de épocas
          "PATIENCE": 50, # Número de épocas sin mejora para terminal el entrenamiento
          'BACKBONE': 'shufflenet_v2_x0_5', # Aruitectura a utilizar como base para Faster R-CNN
          'FPN': False, # Activar uso o no de FPN
          'ANCHOR_SIZE': ((32, 64, 128,256),), # Tamaños de las Cajas Acla
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),), # Relaciones de aspectod e als cajas ancla
          'MIN_SIZE': 512, # Tamaño mínimo de las imágenes
          'MAX_SIZE': 512, # Tamaño máximo de las  imágenes
          'IMG_MEAN': [0.485, 0.456, 0.406], # Medias de ImageNet (Donde se preentrenaron los modelos)
          'IMG_STD': [0.229, 0.224, 0.225], # Desviaciones estándar de ImageNet (Donde se preentrenaron los modelos)
          'IOU_THRESHOLD': 0.5 # Umbral de Intersección sobre Union para evaluar predicciones en entrenamiento
          }


def main():
    ## -- Configuraciones y Carga de datos --##
    # Llave personal de usuario obtenida de Neptune.ai
    # Se puede copiar y poner directamente la llave.
    # api_key = os.getenv("NEPTUNE")  # Si se corre asi, se necesita configurar la clave como una variable de entorno
    api_key = str(sys.argv[1])
    # Crear y obtener el directorio para guardar los checkpoints
    save_dir = os.getcwd() if not params["SAVE_DIR"] else params["SAVE_DIR"]

    # Directorio de la raiz de los datos
    root = pathlib.Path("../data/ChestXRay8/512")

    # Archivos de entrada (imágenes) y objetivos (etiquetas)
    inputs = get_filenames_of_path(root / 'ChestBBImages')
    targets = get_filenames_of_path(root / 'ChestBBLabels')

    # Ordenar entradas y objtivos
    inputs.sort()
    targets.sort()

    # Mapear las etiquetas con valores enteros
    mapping = read_json(pathlib.Path('LabelsMappping.json'))
    mapping

    ## -- Définición de las transformaciones para cada conjunto --##
    # Transformaciones iniciales de entrenameinto (formato, normalizacion a media 0 y std 1)
    # Aumentado con volteos y rescalados
    transforms_training = ComposeDouble([
            Clip(),
            # AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
            # AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
            # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
            # FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
            RescaleWithBB([512],'bilinear')
         ])

    # Transformaciones para validación (formato, normalizacion a media 0 y std 1)
    transforms_validation = ComposeDouble([
            Clip(),
            # FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
            RescaleWithBB([512],'bilinear')
         ])

    # Transformaciones para datos de prueba (formato, normalizacion a media 0 y std 1)
    transforms_test = ComposeDouble([
            Clip(),
            # FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
            RescaleWithBB([512],'bilinear')
         ])

    # Semilla de aleatoreidad
    # seed_everything(params["SEED"])

    ## -- Division del conjunto de datos en subconjuntos (entrenamiento, validación y prueba) --##
    # Parrticipación estratificada: misma cantidad de instancias respecto a sus etiquetas en cada subconjunto
    StratifiedPartition = read_json(pathlib.Path('../DatasetSplits/ChestXRay8/split1.json'))

    inputs_train = [pathlib.Path('../data/ChestXRay8/512/ChestBBImages/' + i[:-4] + '.png') for i in list(StratifiedPartition['Train'].keys())]
    targets_train = [pathlib.Path('../data/ChestXRay8/512/ChestBBLabels/' + i[:-4] + '.json') for i in list(StratifiedPartition['Train'].keys())]

    inputs_valid = [pathlib.Path('../data/ChestXRay8/512/ChestBBImages/' + i[:-4] + '.png') for i in list(StratifiedPartition['Val'].keys())]
    targets_valid = [pathlib.Path('../data/ChestXRay8/512/ChestBBLabels/' + i[:-4] + '.json') for i in list(StratifiedPartition['Val'].keys())]

    inputs_test = [pathlib.Path('../data/ChestXRay8/512/ChestBBImages/' + i[:-4] + '.png') for i in list(StratifiedPartition['Test'].keys())]
    targets_test = [pathlib.Path('../data/ChestXRay8/512/ChestBBLabels/' + i[:-4] + '.json') for i in list(StratifiedPartition['Test'].keys())]

    lt = len(inputs_train)+len(inputs_valid)+len(inputs_test)
    ltr,ptr,lvd,pvd,lts,pts = len(inputs_train), len(inputs_train)/lt, len(inputs_valid), len(inputs_valid)/lt, len(inputs_test), len(inputs_test)/lt
    print('Total de datos: {}\nDatos entrenamiento: {} ({:.2f}%)\nDatos validación: {} ({:.2f}%)\nDatos Prueba: {} ({:.2f}%)'.format(lt,ltr,ptr,lvd,pvd,lts,pts))

    ## -- Creación y visualización de los conjuntos de datos --##
    # Crear conjunto de datos de entrenamiento
    dataset_train = ObjectDetectionDataSet(inputs=inputs_train,
                                           targets=targets_train,
                                           transform=transforms_training,
                                           add_dim = 3,
                                           use_cache=True,
                                           convert_to_format=None,
                                           mapping=mapping,
                                           tgt_int64=True)

    # Crear conjunto de datos de validación
    dataset_valid = ObjectDetectionDataSet(inputs=inputs_valid,
                                           targets=targets_valid,
                                           transform=transforms_validation,
                                           add_dim = 3,
                                           use_cache=True,
                                           convert_to_format=None,
                                           mapping=mapping,
                                           tgt_int64=True)

    # Crear conjunto de datos de prueba
    dataset_test = ObjectDetectionDataSet(inputs=inputs_test,
                                          targets=targets_test,
                                          transform=transforms_test,
                                          add_dim = 3,
                                          use_cache=True,
                                          convert_to_format=None,
                                          mapping=mapping,
                                          tgt_int64=True)

    ## -- Creación de los cargadores de datos --##
    # Crear cargador de datos de entrenamiento
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=params['BATCH_SIZE'],
                                  shuffle=True,
                                  num_workers=6,
                                  collate_fn=collate_double)

    # Crear cargador de datos de validacion
    dataloader_valid = DataLoader(dataset=dataset_valid,
                                  batch_size=params['BATCH_SIZE'],
                                  shuffle=False,
                                  num_workers=6,
                                  collate_fn=collate_double)

    # Crear cargador de datos de prueba
    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=params['BATCH_SIZE'],
                                 shuffle=False,
                                 num_workers=6,
                                 collate_fn=collate_double)

    ## -- Preparación de entorno para correr modelo --##
    #Cargador a Neptune
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project_name=f'{params["OWNER"]}/{params["PROJECT"]}',
        experiment_name=params['EXPERIMENT'],
        params=params
    )

    assert neptune_logger.name  # Se obtiene una solicitud http para verificar la existencia del proyecto en neptune

    # Inicializar el modelo
    model = get_fasterRCNN_shufflenet_v2(num_classes=params['CLASSES'], ## get_fasterRCNN_resnet, get_fasterRCNN_mobilenet, get_fasterRCNN_shufflenet_v2, get_fasterRCNN_efficientnet
                                         backbone_name= params['BACKBONE'],
                                         anchor_size=params['ANCHOR_SIZE'],
                                         aspect_ratios=params['ASPECT_RATIOS'],
                                         fpn=params['FPN'],
                                         min_size=params['MIN_SIZE'],
                                         max_size=params['MAX_SIZE'])

    ## -- Preparar entrenamiento --##
    # Inicializador de Pytorch Lightning
    task = FasterRCNN_lightning(model=model, lr=params['LR'], iou_threshold=params['IOU_THRESHOLD'], torch_mets = [['macro','micro'],'global', True])

    # Monitoreos
    checkpoint_callback = ModelCheckpoint(monitor='Validation_mAP', mode='max')
    learningrate_callback = LearningRateMonitor(logging_interval='step', log_momentum=False)
    early_stopping_callback = EarlyStopping(monitor='Validation_mAP', patience=200, mode='max')

    # Inicializador del entrenamiento
    trainer = Trainer(gpus=params["GPU"],
                      precision=params['PRECISION'],  # Al probar con 16, enable_pl_optimizer=False
                      callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
                      default_root_dir=save_dir,  # Directorio para guardar los checkpoints
                      logger=neptune_logger,
                      log_every_n_steps=1,
                      num_sanity_val_steps=0,
                      benchmark = True#,
                      #accumulate_grad_batches=4#,  # Tambien se puede diccionario para modificar el numero de accumulated batches en cada epoca {indexEpoch:Num.Acc.Batches}
                      # enable_pl_optimizer=False,  # Se descomenta cuando se usa precisión de 16
                      )

    ## -- Ejecutar entrenamiento --##
    # Comenzar el entrenamiento-validación
    trainer.max_epochs = params["MAXEPOCHS"]
    trainer.fit(task, train_dataloader=dataloader_train, val_dataloaders=dataloader_valid)

    ## -- Prueba post-entrenamiento y Carga de datos a Neptune.ai --##
    ## Obtener el mejor modelo y usarlo para predecir la información de prueba,
    # basado en el conjunto de datos de validación y conforme a la metrica usada (mAP from pascal VOC)
    # Realizar evaluación con el cubconjunto de prueba
    trainer.test(ckpt_path="best", test_dataloaders=dataloader_test)

    # Cargar los paquetes utilizados a neptune
    log_packages_neptune(neptune_logger)

    # Cargar el mapeo de clases con valores enteros a neptune
    log_mapping_neptune(mapping, neptune_logger)

    # Cargar el modelo a neptune
    if params["LOG_MODEL"]:
        checkpoint_path = pathlib.Path(checkpoint_callback.best_model_path)
        log_model_neptune(checkpoint_path=checkpoint_path,
                          save_directory=pathlib.Path.home(),
                          name="best_model.pt",
                          neptune_logger=neptune_logger)

    # Parar el cargador
    neptune_logger.experiment.stop()
    print("Finished")


if __name__ == "__main__":
    main()
