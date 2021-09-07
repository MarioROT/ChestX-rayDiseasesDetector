from typing import Tuple

import torch
from torch import nn
from torch.jit.annotations import List, Optional, Dict
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class AnchorGenerator(nn.Module):
    # Adaptado del módulo AnchorGenerator de torchvision.
    # Regresa anchors_over_all_feature_maps.

    """
    Módulo que genera "anclas" para un conjunto de mapas de
    características y tamaños de imágenes.

    Este módulo apoya el cómputo de cajas ancla de múltiples
    tamaños y relaciones de aspecto por mapa de características.
    Este módulo asume la relación de aspecto  = altura / anchura
    por cada caja ancla.

    Las variables sizes y aspect_ratios deberían tener el mismo número de elementos,
    y debe corresponder al mismo tiempo con el número de mapas de características.

    Por lo tanto sizes[i] y aspect_ratios[i] pueden tener un número arbitrario de elementos,
    y AnchorGenerator generará como salida un conjunto de sizes[i] * aspect_ratios[i]
    cajas anclas por ubicación espacial para cada mapa de características 'i'.

    Argumentos:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]],
    }

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super(AnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(
        self, scales, aspect_ratios, dtype=torch.float32, device="cpu"
    ):
        # tipo: (List[int], List[float], int, Device) -> Tensor  # noqa: F821
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        # Tipo: (int, Device) -> None  # noqa: F821
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device == device:
                return

        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # Para cada combinación (a, (g, s), i) en (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # Salida g[i] cajas ancla que son s[i] distanciado aparte en dirección i, con la misma dimensión que a.
    def grid_anchors(self, grid_sizes, strides):
        # tipo: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        assert len(grid_sizes) == len(strides) == len(cell_anchors)

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # Para la caja ancla de salida, se calcula [x_center, y_center, x_center, y_center]
            shifts_x = (
                torch.arange(0, grid_width, dtype=torch.float32, device=device)
                * stride_width
            )
            shifts_y = (
                torch.arange(0, grid_height, dtype=torch.float32, device=device)
                * stride_height
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # Para cada par (base anchor, output anchor),
            # se desplaza cada caja ancla base centrada en cero
            # por el centro de la caja ancla de salida
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        # tipo: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # tipo: (ImageList, List[Tensor]) -> List[Tensor]
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device),
            ]
            for g in grid_sizes
        ]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        self._cache.clear()
        return anchors_over_all_feature_maps


def get_anchor_boxes(
    image: torch.tensor,
    rcnn_transform: GeneralizedRCNNTransform,
    feature_map_size: tuple,
    anchor_size: Tuple[tuple] = ((128, 256, 512),),
    aspect_ratios: Tuple[tuple] = ((1.0,),),
):
    """
    Regresa las cajas anclas para una imagen dada y un mapa
    de características. El argumento de entrada 'image' debiere
    ser tipo torch.tensor con forma [C, H, W]. El argumento de
    entrada feature_map_size debiere ser una tupla con forma [C, H, W].

    Ejemplo de uso:

    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    transform = GeneralizedRCNNTransform(min_size=1024,
                                         max_size=1024,
                                         image_mean=[0.485, 0.456, 0.406],
                                         image_std=[0.229, 0.224, 0.225])

    image = dataset[0]['x'] # ObjectDetectionDataSet

    anchors = get_anchor_boxes(image,
                               transform,
                               feature_map_size=(512, 16, 16),
                               anchor_size=((128, 256, 512),),
                               aspect_ratios=((1.0, 2.0),)
                               )
    """

    image_transformed = rcnn_transform([image])

    features = [torch.rand(size=feature_map_size)]

    anchor_gen = AnchorGenerator(anchor_size, aspect_ratios)
    anchors = anchor_gen(image_list=image_transformed[0], feature_maps=features)

    return anchors[0]
