import pathlib
from multiprocessing import Pool
from typing import List, Dict

import torch
from skimage.color import rgba2rgb
from skimage.io import imread
from torchvision.ops import box_convert

from transformations import ComposeDouble, ComposeSingle
from transformations import map_class_to_int
from utils import read_json, read_pt
import numpy as np


class ObjectDetectionDataSet(torch.utils.data.Dataset):
    """
    Construye un conjunto de datos con imágenes y sus respectivas etiquetas (objetivos).
    Cada target es esperado que se encuentre en un archivo JSON individual y debe contener
    al menos las llaves 'boxes' y 'labels'.
    Las entradas (imágenes) y objetivos (etiquetas) son esperadas como una lista de
    objetos pathlib.Path

    En caso de que las etiquetas esten en formato string, puedes usar un diccionario de
    mapeo para codificarlas como enteros (int).

    Regresa un diccionario con las siguientes llaves: 'x', 'y'->('boxes','labels'), 'x_name', 'y_name'
    """

    def __init__(
        self,
        inputs: List[pathlib.Path],
        targets: List[pathlib.Path],
        transform: ComposeDouble = None,
        add_dim: bool = False,
        use_cache: bool = False,
        convert_to_format: str = None,
        mapping: Dict = None,
        tgt_int64: bool = False,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.add_dim = add_dim
        self.use_cache = use_cache
        self.convert_to_format = convert_to_format
        self.mapping = mapping
        self.tgt_int64 = tgt_int64

        if self.use_cache:
            # Usar multiprocesamiento para cargar las imagenes y las etiquetas en la memoria RAM
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets))


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Seleccionar una muestra
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Cargar entradas (imágenes) y objetivos (etiquetas)
            x, y = self.read_images(input_ID, target_ID)

        # De RGBA a RGB
        if x.shape[-1] == 4:
            x = rgba2rgb(x)

        # Leer cajas
        try:
            boxes = torch.from_numpy(y["boxes"]).to(torch.float32)
        except TypeError:
            boxes = torch.tensor(y["boxes"]).to(torch.float32)

        # Leer puntajes
        if "scores" in y.keys():
            try:
                scores = torch.from_numpy(y["scores"]).to(torch.float32)
            except TypeError:
                scores = torch.tensor(y["scores"]).to(torch.float32)

        # Mapeo de etiquetas
        if self.mapping:
            labels = map_class_to_int(y["labels"], mapping=self.mapping)
        else:
            labels = y["labels"]

        # Leer etiquetas
        try:
            labels = torch.from_numpy(labels).to(torch.int64)
        except TypeError:
            labels = torch.tensor(labels).to(torch.int64)

        # Convertir formato
        if self.convert_to_format == "xyxy":
            boxes = box_convert(
                boxes, in_fmt="xywh", out_fmt="xyxy"
            )  # Transformaciones de las cajas del formato xywh a xyxy
        elif self.convert_to_format == "xywh":
            boxes = box_convert(
                boxes, in_fmt="xyxy", out_fmt="xywh"
            )  # # Transformaciones de las cajas del formato xyxy a xywh

        # Crear objetivos
        target = {"boxes": boxes, "labels": labels}

        if "scores" in y.keys():
            target["scores"] = scores

        # Preprocesamiento
        target = {
            key: value.numpy() for key, value in target.items()
        }  # Todos los tensores debieren ser convertidos a np.ndarrays

        if self.transform is not None:
            x, target = self.transform(x, target)  # Regresa np.ndarrays

        if "scores" in y.keys():
            bxs,lbs,srs = [],[],[]
            for r,f in enumerate(target['scores']):
                if f > 0.70:
                    bxs.append(target['boxes'][r])
                    lbs.append(target['labels'][r])
                    srs.append(target['scores'][r])
            target = {'boxes':np.array(bxs), 'labels':np.array(lbs), 'scores':np.array(srs)}

        if self.add_dim:
            if len(x.shape) == 2:
                x = x.T
                x = np.array([x])
            # print(x.shape)
            # x = np.moveaxis(x, source=1, destination=-1)
            # x = np.expand_dims(x, axis=0)

        # print('Before: ', target)
        # Encasillar
        if self.tgt_int64:
            x = torch.from_numpy(x).type(torch.float32)
            target = {
                key: torch.from_numpy(value).type(torch.int64)
                for key, value in target.items()
            }
        else:
            x = torch.from_numpy(x).type(torch.float32)
            target = {
                key: torch.from_numpy(value).type(torch.float64)#int64)
                for key, value in target.items()
            }
        # print('After: ', target)
        return {
            "x": x,
            "y": target,
            "x_name": self.inputs[index].name,
            "y_name": self.targets[index].name,
        }

    @staticmethod
    def read_images(inp, tar):
        return imread(inp), read_json(tar) #read_pt(tar)


class ObjectDetectionDatasetSingle(torch.utils.data.Dataset):
    """
    Construir un conjunto de datos únicamente con imágenes
    Las entradas se espera que sean una lista de objetos de pathlib.Path.

    Regresa un diccionario con las llaves: 'x', 'x_name'
    """

    def __init__(
        self,
        inputs: List[pathlib.Path],
        transform: ComposeSingle = None,
        use_cache: bool = False,
    ):
        self.inputs = inputs
        self.transform = transform
        self.use_cache = use_cache

        if self.use_cache:
            # Usar multiprocesamiento para cargar las imagenes y las etiquetas en la memoria RAM
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x = self.cached_data[index]
        else:
            # Seleccionar una muestra
            input_ID = self.inputs[index]

            # Cargar entrada (imagen) y objetivo (etiqueta)
            x = self.read_images(input_ID)

        # De RGBA a RGB
        if x.shape[-1] == 4:
            x = rgba2rgb(x)

        # Preprocesamiento
        if self.transform is not None:
            x = self.transform(x)  # regresa a np.ndarray

        # Encasillar
        x = torch.from_numpy(x).type(torch.float32)

        return {"x": x, "x_name": self.inputs[index].name}

    @staticmethod
    def read_images(inp):
        return imread(inp)
