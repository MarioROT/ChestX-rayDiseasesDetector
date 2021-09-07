import os
import pathlib

import albumentations as A
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers.neptune import NeptuneLogger
from torch.utils.data import DataLoader

from datasets import ObjectDetectionDataSet
from faster_RCNN import FasterRCNN_lightning
from faster_RCNN import get_fasterRCNN_resnet
from transformations import Clip, ComposeDouble
from transformations import AlbumentationWrapper
from transformations import FunctionWrapperDouble
from transformations import normalize_01
from utils import get_filenames_of_path, collate_double
from utils import log_mapping_neptune
from utils import log_model_neptune
from utils import log_packages_neptune

# Hiperparámetros
params = {
    'BATCH_SIZE': 4, # Tamaño del lote
    'OWNER': 'rubsini',  # Nombre de usuario en Neptune.ai
    'SAVE_DIR': "../Experiments/",  # Directorio para guardar los checkpoints en entrenamiento
    'LOG_MODEL': False,  # Si se cargará el modelo a neptune después del entrenamiento
    'GPU': 1,  # Activar o  desactivar para elegir entrenar en GPU o CPU
    'LR': 0.001, # Tasa de aprendizaje
    'PRECISION': 32, # Precisión de cálculo
    'CLASSES': 9, # Número de clases (incluyendo el fondo)
    'SEED': 42, # Semilla de aleatoreidad
    'PROJECT': 'DiseasesDetection', # Nombre dle proyecto creado en Neptune.ai
    'EXPERIMENT': 'chests', # nombre del experimento dentro del proyecto
    'MAXEPOCHS': 500, # Número máximo de épocas
    "PATIENCE": 50, # Número de épocas sin mejora para terminal el entrenamiento
    'BACKBONE': 'resnet18', # Aruitectura a utilizar como base para Faster R-CNN
    'FPN': False, # Activar uso o no de FPN
    'ANCHOR_SIZE': ((32, 64, 128, 256, 512),), # Tamaños de las Cajas Acla
    'ASPECT_RATIOS': ((0.5, 1.0, 2.0),), # Relaciones de aspectod e als cajas ancla
    'MIN_SIZE': 1024, # Tamaño mínimo de las imágenes
    'MAX_SIZE': 1024, # Tamaño máximo de las  imágenes
    'IMG_MEAN': [0.485, 0.456, 0.406], # Medias de ImageNet (Donde se preentrenaron los modelos)
    'IMG_STD': [0.229, 0.224, 0.225], # Desviaciones estándar de ImageNet (Donde se preentrenaron los modelos)
    'IOU_THRESHOLD': 0.5 # Umbral de Intersección sobre Union para evaluar predicciones en entrenamiento
}


def main():
    # Llave personal de usuario obtenida de Neptune.ai
    # Se puede copiar y poner directamente la llave.
    api_key = os.getenv("NEPTUNE")  # Si se corre asi, se necesita configurar la clave como una variable de entorno

    # Guardar directorio
    save_dir = os.getcwd() if not params["SAVE_DIR"] else params["SAVE_DIR"]

    # Directorio de la raiz de los datos
    root = pathlib.Path("../data/ChestXRay8")

    # Archivos de entrada (imágenes) y objetivos (etiquetas)
    inputs = get_filenames_of_path(root / "input")
    targets = get_filenames_of_path(root / "target")

    inputs.sort()
    targets.sort()

    # Mapear las etiquetas con valores enteros
    mapping = read_json(pathlib.Path('LabelsMappping.json'))
    mapping

    # Transformaciones iniciales de entrenameinto (formato, normalizacion a media 0 y std 1)
    # Aumentado con volteos y rescalados
    transforms_training = ComposeDouble(
        [
            Clip(),
            # AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
            # AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
            # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
        ]
    )

    # Transformaciones para validación (formato, normalizacion a media 0 y std 1)
    transforms_validation = ComposeDouble(
        [
            Clip(),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
        ]
    )

    # Transformaciones para datos de prueba (formato, normalizacion a media 0 y std 1)
    transforms_test = ComposeDouble(
        [
            Clip(),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
        ]
    )

    # Semilla de aleatoreidad
    seed_everything(params["SEED"])

    # Division de los datos en subconjuntos (entrenamiento, validación y prueba)
    inputs_train, inputs_valid, inputs_test = inputs[:int(len(inputs)*0.7)], inputs[int(len(inputs)*0.7):int(len(inputs)*0.8)], inputs[int(len(inputs)*0.8):]
    targets_train, targets_valid, targets_test = targets[:int(len(inputs)*0.7)], targets[int(len(inputs)*0.7):int(len(inputs)*0.8)], targets[int(len(inputs)*0.8):]

    # Crear conjunto de datos de entrenamiento
    dataset_train = ObjectDetectionDataSet(inputs=inputs_train,
                                           targets=targets_train,
                                           transform=transforms_training,
                                           add_dim = True,
                                           use_cache=True,
                                           convert_to_format=None,
                                           mapping=mapping,
                                           tgt_int64=True)

    # Crear conjunto de datos de validación
    dataset_valid = ObjectDetectionDataSet(inputs=inputs_valid,
                                           targets=targets_valid,
                                           transform=transforms_validation,
                                           add_dim = True,
                                           use_cache=True,
                                           convert_to_format=None,
                                           mapping=mapping,
                                           tgt_int64=True)

    # Crear conjunto de datos de prueba
    dataset_test = ObjectDetectionDataSet(inputs=inputs_test,
                                          targets=targets_test,
                                          transform=transforms_test,
                                          add_dim = True,
                                          use_cache=True,
                                          convert_to_format=None,
                                          mapping=mapping,
                                          tgt_int64=True)

    # Crear cargador de datos de entrenamiento
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=params['BATCH_SIZE'],
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=collate_double)

    # Crear cargador de datos de validacion
    dataloader_valid = DataLoader(dataset=dataset_valid,
                                  batch_size=params['BATCH_SIZE'],
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=collate_double)

    # Crear cargador de datos de prueba
    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=2,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=collate_double)

    #Cargador a Neptune
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project_name=f'{params["OWNER"]}/{params["PROJECT"]}',
        experiment_name=params['EXPERIMENT'],
        params=params
    )

    assert neptune_logger.name  # Se obtiene una solicitud http para verificar la existencia del proyecto en neptune

    # Inicializar el modelo
    model = get_fasterRCNN_resnet(
        num_classes=params["CLASSES"],
        backbone_name=params["BACKBONE"],
        anchor_size=params["ANCHOR_SIZE"],
        aspect_ratios=params["ASPECT_RATIOS"],
        fpn=params["FPN"],
        min_size=params["MIN_SIZE"],
        max_size=params["MAX_SIZE"],
    )

    # Inicializador de Pytorch Lightning
    task = FasterRCNN_lightning(
        model=model, lr=params["LR"], iou_threshold=params["IOU_THRESHOLD"]
    )

    # Monitoreos
    checkpoint_callback = ModelCheckpoint(monitor='Validation_mAP', mode='max')
    learningrate_callback = LearningRateMonitor(logging_interval='step', log_momentum=False)
    early_stopping_callback = EarlyStopping(monitor='Validation_mAP', patience=50, mode='max')

    # Inicializador del entrenamiento
    trainer = Trainer(
        gpus=params["GPU"],
        precision=params["PRECISION"],  # Probar 16 con enable_pl_optimizer=False
        callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
        default_root_dir=save_dir,  # Donde será guardados los checkpoints
        logger=neptune_logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    # Comenzar el entrenamiento-validación
    trainer.max_epochs = params["MAXEPOCHS"]
    trainer.fit(
        task, train_dataloader=dataloader_train, val_dataloaders=dataloader_valid
    )

    # Realizar evaluación con el cubconjunto de prueba
    trainer.test(ckpt_path="best", test_dataloaders=dataloader_test)

    # Cargar los paquetes utilizados a neptune
    log_packages_neptune(neptune_logger)

    # Cargar el mapeo de clases con valores enteros a neptune
    log_mapping_neptune(mapping, neptune_logger)

    # Cargar el modelo a neptune
    if params["LOG_MODEL"]:
        checkpoint_path = pathlib.Path(checkpoint_callback.best_model_path)
        log_model_neptune(
            checkpoint_path=checkpoint_path,
            save_directory=pathlib.Path.home(),
            name="best_model.pt",
            neptune_logger=neptune_logger,
        )

    # Parar el cargador
    neptune_logger.experiment.stop()
    print("Finished")


if __name__ == "__main__":
    main()
