import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple

import napari
import numpy as np
import torch
from magicgui.widgets import Combobox, Slider
from magicgui.widgets import FloatSlider, Container, Label
from napari.layers import Shapes
from skimage.io import imread
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import box_convert
from torchvision.ops import nms

from anchor_generator import get_anchor_boxes
from datasets import ObjectDetectionDataSet
from datasets import ObjectDetectionDatasetSingle
from transformations import re_normalize
from utils import color_mapping_func
from utils import enable_gui_qt
from utils import read_json, save_json


def make_bbox_napari(bbox, reverse=False):
    """
    Obtener las coordenadas de las cuatro esquinas de una caja delimitadora,
    se espera que sea en formato 'xyxy'.

    El resultado puede ser puesto directamente en las capas de formas de napari.

    Orden: arriba-izquierda, abajo-izquierda, abajo-derecha, arriba-derecha
    estilo numpy ---> [y, x]

    """
    if reverse:
        x = bbox[:, 1]
        y = bbox[:, 0]

        x1 = x.min()
        y1 = y.min()
        x2 = x.max()
        y2 = y.max()

        return np.array([x1, y1, x2, y2])

    else:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        bbox_rect = np.array([[y1, x1], [y2, x1], [y2, x2], [y1, x2]])
        return bbox_rect


def get_center_bounding_box(boxes: torch.tensor):
    """Regresa los puntos centrales de una caja delimitadora dada."""
    return box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")[:, :2]


class ViewerBase:
    def napari(self):
        # IPython magic para napari < 0.4.8
        enable_gui_qt()

        # napari
        if self.viewer:
            try:
                del self.viewer
            except AttributeError:
                pass
        self.index = 0

        # Iniciar una instancia napari
        self.viewer = napari.Viewer()

        # Mostrar la muestra actual
        self.show_sample()

        # Comandos de teclado
        # Presionar 'n' para pasar a la siguiente muestra
        @self.viewer.bind_key("n")
        def next(viewer):
            self.increase_index()  # Incrementar el índice
            self.show_sample()  # Mostrar la siguiente muestra

        # Presionar 'b' para regresar a la muestra anterior
        @self.viewer.bind_key("b")
        def prev(viewer):
            self.decrease_index()  # Decrementar el ínidce
            self.show_sample()  # Mostrar la siguiente muestra

    def increase_index(self):
        self.index += 1
        if self.index >= len(self.dataset):
            self.index = 0

    def decrease_index(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.dataset) - 1

    def show_sample(self):
        """Método de sobrescritura"""
        pass

    def create_image_layer(self, x, x_name):
        return self.viewer.add_image(x, name=str(x_name))

    def update_image_layer(self, image_layer, x, x_name):
        """Reemplazar la información y el nombre de una image_layer dada"""
        image_layer.data = x
        image_layer.name = str(x_name)

    def get_all_shape_layers(self):
        return [layer for layer in self.viewer.layers if isinstance(layer, Shapes)]

    def remove_all_shape_layers(self):
        all_shape_layers = self.get_all_shape_layers()
        for shape_layer in all_shape_layers:
            self.remove_layer(shape_layer)

    def remove_layer(self, layer):
        self.viewer.layers.remove(layer)


class DatasetViewer(ViewerBase):
    def __init__(
        self,
        dataset: ObjectDetectionDataSet,
        color_mapping: Dict,
        rccn_transform: GeneralizedRCNNTransform = None,
    ):
        self.dataset = dataset
        self.index = 0
        self.color_mapping = color_mapping

        # Visor de instancia napari
        self.viewer = None

        # RCNN_transformer
        self.rccn_transform = rccn_transform

        # imagen y capa de forma actual
        self.image_layer = None
        self.shape_layer = None

    def show_sample(self):

        # Obtener una muestra del dataset
        sample = self.get_sample_dataset(self.index)

        # RCNN-transformer
        if self.rccn_transform is not None:
            sample = self.rcnn_transformer(sample, self.rccn_transform)

        # Transformar una muestra a numpy, CPU y el formato correcto a visualizar
        x, x_name = self.transform_x(sample)
        y, y_name = self.transform_y(sample)

        # Crear una capa de imagen
        if self.image_layer not in self.viewer.layers:
            self.image_layer = self.create_image_layer(x, x_name)
        else:
            self.update_image_layer(self.image_layer, x, x_name)

        # Crear una capa de forma
        if self.shape_layer not in self.viewer.layers:
            self.shape_layer = self.create_shape_layer(y, y_name)
        else:
            self.update_shape_layer(self.shape_layer, y, y_name)

        # Reiniciar vista
        self.viewer.reset_view()

        # self.viewer.layers.select_previous()  # enfocar en una capa de entrada
        # self.viewer.status = f'index: {self.index}, x_name: {x_name}, y_name: {y_name}'

    def get_sample_dataset(self, index):
        return self.dataset[index]

    def transform_x(self, sample):
        # desempaquetar diccionario
        x, x_name = sample["x"], sample["x_name"]

        # Asegurarse de que es numpy.ndarray en el CPU
        x = x.cpu().numpy()

        # De [C, H, W] a [H, W, C] - solo para imágenes en RGB.
        # if self.check_if_rgb(x):
        #     x = np.moveaxis(x, source=0, destination=-1)
        if len(x.shape) == 2:
            x = x.T
            x = x[np.newaxis,...]
        # x = x.T  # Para parasrlas de [W,H] a [H,W]
        # x = x[..., np.newaxis] # Añadido para imagenes de un canal
        # print(len(x.shape))
        # Re-normalizar
        x = re_normalize(x)

        return x, x_name

    def transform_y(self, sample):
        # Desempaquetar diccionario
        y, y_name = sample["y"], sample["y_name"]

        # Asegurarse de que es numpy.ndarray en el CPU
        y = {key: value.cpu().numpy() for key, value in y.items()}

        return y, y_name

    def get_boxes(self, y):
        boxes = y["boxes"]

        # Transformar cajas delimitadoras para hacerlas compatibles con napari
        boxes_napari = [make_bbox_napari(box) for box in boxes]

        return boxes_napari

    def get_labels(self, y):
        return y["labels"]

    def get_colors(self, y):
        return color_mapping_func(y["labels"], self.color_mapping)

    def get_scores(self, y):
        return y["scores"]

    def get_text_parameters(self):
        return {
            "text": "{labels}",
            "size": 10,
            "color": "white",
            "anchor": "upper_left",
            "translation": [-1, 0],
        }

    def create_shape_layer(self, y, y_name):
        boxes = self.get_boxes(y)
        labels = self.get_labels(y)
        colors = self.get_colors(y)

        # Añadir propiedades a la capa de forma
        # Esto se requiere para obtener el txto correcto para el TextManager
        # El TextManager  muestra el texto en la parte superior de la caja delimitadora
        # en este caso es la etiqueta atribuida  acada caja delimitadora

        text_parameters = self.get_text_parameters()  # diccionario
        properties = {"labels": labels}

        if "scores" in y.keys():
            scores = self.get_scores(y)
            text_parameters["text"] = "label: {labels}\nscore: {scores:.2f}"
            properties["scores"] = scores

        # Añadir una capa de forma
        shape_layer = self.viewer.add_shapes(
            data=boxes,
            face_color="transparent",
            edge_color="red",
            edge_width=2,
            properties=properties,
            name=y_name,
            text=text_parameters,
        )

        # Convertir la capa en no-editable
        shape_layer.editable = False

        # Guardar información como metadatos
        self.save_to_metadata(shape_layer, "boxes", boxes)
        self.save_to_metadata(shape_layer, "labels", labels)
        self.save_to_metadata(shape_layer, "colors", colors)

        # Añadir puntajes
        if "scores" in y.keys():
            self.save_to_metadata(shape_layer, "scores", scores)

        # Actualizar Color.
        self.set_colors_of_shapes(shape_layer, self.color_mapping)

        return shape_layer

    def update_shape_layer(self, shape_layer, y, y_name):
        """Remove all shapes and replace the data and the properties"""
        # Eliminar todas las foras de una capa
        self.select_all_shapes(shape_layer)
        shape_layer.remove_selected()

        boxes = self.get_boxes(y)
        labels = self.get_labels(y)
        colors = self.get_colors(y)

        if "scores" in y.keys():
            scores = self.get_scores(y)

        # Configurar las propiedades actuales
        shape_layer.current_properties["labels"] = labels
        if "scores" in y.keys():
            shape_layer.current_properties["scores"] = scores

        # Añadir formas a la capa
        shape_layer.add(boxes)

        # Configurar las propuedades de dorma correcta
        shape_layer.properties["labels"] = labels
        if "scores" in y.keys():
            shape_layer.properties["scores"] = scores

        # Anular la información en los metadatos
        self.reset_metadata(shape_layer)
        self.save_to_metadata(shape_layer, "boxes", boxes)
        self.save_to_metadata(shape_layer, "labels", labels)
        self.save_to_metadata(shape_layer, "colors", colors)

        # Añadir puntajes
        if "scores" in y.keys():
            self.save_to_metadata(shape_layer, "scores", scores)

        # Actualizar color
        self.set_colors_of_shapes(shape_layer, self.color_mapping)

        # Cambiar el nombre
        shape_layer.name = y_name

    def save_to_metadata(self, shape_layer, key, value):
        shape_layer.metadata[key] = value

    def reset_metadata(self, shape_layer):
        shape_layer.metadata = {}

    def check_if_rgb(self, x):
        """Verificar si la primer dimensión de la imagen es el número de canales, y es 3"""
        # TODO: Las imágenes RGBA tienen 4 canles -> se genera Error
        if x.shape[0] == 3:
            return True
        else:
            raise AssertionError(
                f"The channel dimension is supposed to be 3 for RGB images. This image has a channel dimension of size {x.shape[0]}"
            )

    def get_unique_labels(self, shapes_layer):
        return set(shapes_layer.metadata["labels"])

    def select_all_shapes(self, shape_layer):
        """Seleciona todas las formas dentro de una instancia shape_layer."""
        shape_layer.selected_data = set(range(shape_layer.nshapes))

    def select_all_shapes_label(self, shape_layer, label):
        """Selecciona todas las formas de una determinada etiqueta"""
        if label not in self.get_unique_labels(shape_layer):
            raise ValueError(
                f"Label {label} does not exist. Available labels are {self.get_unique_labels(shape_layer)}!"
            )

        indices = set(self.get_indices_of_shapes(shape_layer, label))
        shape_layer.selected_data = indices

    def get_indices_of_shapes(self, shapes_layer, label):
        return list(np.argwhere(shapes_layer.properties["labels"] == label).flatten())

    def set_colors_of_shapes(self, shape_layer, color_mapping):
        """ Itera sobre etiquetas únicas y asigna un color conforme a el color_mapping."""
        for label in self.get_unique_labels(shape_layer):  # get unique labels
            color = color_mapping[label]  # get color from mapping
            self.set_color_of_shapes(shape_layer, label, color)

    def set_color_of_shapes(self, shapes_layer, label, color):
        """Asigna un oclor a cada forma de una determinada etiqueta"""
        self.select_all_shapes_label(
            shapes_layer, label
        )  # Seleccionar únicamente las formas correctas
        shapes_layer.current_edge_color = (
            color  # Cambiar el color de las formas formas seleccionadas
        )

    def gui_text_properties(self, shape_layer):
        container = self.create_gui_text_properties(shape_layer)
        self.viewer.window.add_dock_widget(
            container, name="text_properties", area="right"
        )

    def gui_score_slider(self, shape_layer):
        if "nms_slider" in self.viewer.window._dock_widgets.keys():
            self.remove_gui("nms_slider")
            self.shape_layer.events.name.disconnect(
                callback=self.shape_layer.events.name.callbacks[0]
            )

        container = self.create_gui_score_slider(shape_layer)
        self.slider = container
        self.viewer.window.add_dock_widget(container, name="score_slider", area="right")

    def gui_nms_slider(self, shape_layer):
        if "score_slider" in self.viewer.window._dock_widgets.keys():
            self.remove_gui("score_slider")
            self.shape_layer.events.name.disconnect(
                callback=self.shape_layer.events.name.callbacks[0]
            )

        container = self.create_gui_nms_slider(shape_layer)
        self.slider = container
        self.viewer.window.add_dock_widget(container, name="nms_slider", area="right")

    def remove_gui(self, name):
        widget = self.viewer.window._dock_widgets[name]
        self.viewer.window.remove_dock_widget(widget)

    def create_gui_text_properties(self, shape_layer):
        TextColor = Combobox(
            choices=shape_layer._colors, name="text color", value="white"
        )
        TextSize = Slider(min=1, max=50, name="text size", value=1)

        container = Container(widgets=[TextColor, TextSize])

        def change_text_color(event):
            # Esto cambia el color del texto
            shape_layer.text.color = str(TextColor.value)

        def change_text_size(event):
            # Esto cambia el tamaño del texto
            shape_layer.text.size = int(TextSize.value)

        TextColor.changed.connect(change_text_color)
        TextSize.changed.connect(change_text_size)

        return container

    def create_gui_score_slider(self, shape_layer):
        slider = FloatSlider(min=0.0, max=1.0, step=0.01, name="Score", value=0.0)
        slider_label = Label(name="Score_threshold", value=0.0)

        container = Container(widgets=[slider, slider_label])

        def change_boxes(event, shape_layer=shape_layer):
            # Eliminar todas las formas
            self.select_all_shapes(shape_layer)
            shape_layer.remove_selected()

            # Crear la mascara y nueva información
            mask = np.where(shape_layer.metadata["scores"] > slider.value)
            new_boxes = np.asarray(shape_layer.metadata["boxes"])[mask]
            new_labels = shape_layer.metadata["labels"][mask]
            new_scores = shape_layer.metadata["scores"][mask]

            # Configurar las propiedades actuales
            shape_layer.current_properties["labels"] = new_labels
            shape_layer.current_properties["scores"] = new_scores

            # Añadir formas a una capa
            if new_boxes.size > 0:
                shape_layer.add(list(new_boxes))

            # Configurar las propiedades
            shape_layer.properties["labels"] = new_labels
            shape_layer.properties["scores"] = new_scores

            # Cambiar la etiqueta
            slider_label.value = str(slider.value)

        slider.changed.connect(change_boxes)

        # Invocar puntaje
        container.Score.value = 0.0

        # Evento que se activa cuando el nombre de la capa es cambiado
        self.shape_layer.events.name.connect(change_boxes)

        return container

    def create_gui_nms_slider(self, shape_layer):
        slider = FloatSlider(min=0.0, max=1.0, step=0.01, name="NMS", value=0.0)
        slider_label = Label(name="IoU_threshold")

        container = Container(widgets=[slider, slider_label])

        def change_boxes(event, shape_layer=shape_layer):
            # Remover todas las formas de unas capas
            self.select_all_shapes(shape_layer)
            shape_layer.remove_selected()

            # Crear una mascara y nueva información
            boxes = torch.tensor(
                [
                    make_bbox_napari(box, reverse=True)
                    for box in shape_layer.metadata["boxes"]
                ]
            )
            scores = torch.tensor(shape_layer.metadata["scores"])

            if boxes.size()[0] > 0:
                mask = nms(boxes, scores, slider.value)  # torch.tensor
                mask = (np.array(mask),)

                new_boxes = np.asarray(shape_layer.metadata["boxes"])[mask]
                new_labels = shape_layer.metadata["labels"][mask]
                new_scores = shape_layer.metadata["scores"][mask]

                # Configurar las propiedades
                shape_layer.current_properties["labels"] = new_labels
                shape_layer.current_properties["scores"] = new_scores

                # Añadir formas a una capa
                if new_boxes.size > 0:
                    shape_layer.add(list(new_boxes))

                # Configurar las propiedas
                shape_layer.properties["labels"] = new_labels
                shape_layer.properties["scores"] = new_scores

                # Configurar información temporal
                shape_layer.metadata["boxes_nms"] = list(new_boxes)
                shape_layer.metadata["labels_nms"] = new_labels
                shape_layer.metadata["scores_nms"] = new_scores

            # Cambiar etiqueta
            slider_label.value = str(slider.value)

        slider.changed.connect(change_boxes)

        # Invocar NMS
        container.NMS.value = 1.0

        # Evento lanzado cuando el nombre de las capa de formas cambia
        self.shape_layer.events.name.connect(change_boxes)

        return container

    def rcnn_transformer(self, sample, transform):
        # Desempaquetar diccionario
        x, x_name, y, y_name = (
            sample["x"],
            sample["x_name"],
            sample["y"],
            sample["y_name"],
        )

        x, y = transform([x], [y])
        x, y = x.tensors[0], y[0]

        return {"x": x, "y": y, "x_name": x_name, "y_name": y_name}


class DatasetViewerSingle(DatasetViewer):
    def __init__(
        self,
        dataset: ObjectDetectionDatasetSingle,
        rccn_transform: GeneralizedRCNNTransform = None,
    ):
        self.dataset = dataset
        self.index = 0

        # Instancia del visualizador napari
        self.viewer = None

        # rccn_transformer
        self.rccn_transform = rccn_transform

        # Imagen actual y capa de formase & shape layer
        self.image_layer = None
        self.shape_layer = None

    def show_sample(self):

        # Obtener una muestra del conjunto de datos
        sample = self.get_sample_dataset(self.index)

        # RCNN-transformer
        if self.rccn_transform is not None:
            sample = self.rcnn_transformer(sample, self.rccn_transform)

        # Transformar la muestra a numpy, CPU y el formato correcto a visualizar
        x, x_name = self.transform_x(sample)

        # Crear una capa de imagen
        if self.image_layer not in self.viewer.layers:
            self.image_layer = self.create_image_layer(x, x_name)
        else:
            self.update_image_layer(self.image_layer, x, x_name)

        # Reiniciar vista
        self.viewer.reset_view()

    def rcnn_transformer(self, sample, transform):
        # Desempaquetar diccionario
        x, x_name = sample["x"], sample["x_name"]

        x, _ = transform([x])
        x, _ = x.tensors[0], _

        return {"x": x, "x_name": x_name}


class Annotator(ViewerBase):
    def __init__(
        self,
        image_ids: pathlib.Path,
        annotation_ids: pathlib.Path = None,
        color_mapping: Dict = {},
    ):

        self.image_ids = image_ids
        self.annotation_ids = annotation_ids

        self.index = 0
        self.color_mapping = color_mapping

        # Instancia del visualizador napari
        self.viewer = None

        # Imagen actual y capas de formas
        self.image_layer = None
        self.shape_layers = []

        # Iniciar anotaciones
        self.annotations = self.init_annotations()

        # Cargar anotaciones del disco
        if self.annotation_ids is not None:
            self.load_annotations()

        # Ancho de los bordes para las formas
        self.edge_width = 2.0

        # Aotaciones de los objetos actuales
        self.annotation_object = None

    def init_annotations(self):
        @dataclass
        class AnnotationObject:
            name: str
            boxes: np.ndarray
            labels: np.ndarray

            def __bool__(self):
                return True if self.boxes.size > 0 else False

        return [
            AnnotationObject(
                name=image_id.stem, boxes=np.array([]), labels=np.array([])
            )
            for image_id in self.image_ids
        ]

    def increase_index(self):
        self.index += 1
        if self.index >= len(self.image_ids):
            self.index = 0

    def decrease_index(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.image_ids) - 1

    def show_sample(self):
        # Obtener el identificardor de la imagen
        image_id = self.get_image_id(self.index)

        # Cargar la imagen
        x = self.load_x(image_id)

        # Transformaciones de la imagen
        x = self.transform_x(x)

        # Crear o actualizar una capa de imagen
        if self.image_layer not in self.viewer.layers:
            self.image_layer = self.create_image_layer(x, image_id)
        else:
            self.update_image_layer(self.image_layer, x, image_id)

        # Guardar las anotaciones en annotation_object (cualquier cambio será guardado o sobreescrito)
        self.save_annotations(self.annotation_object)

        # Actualizar el objeto de anotaciones actual
        self.annotation_object = self.get_annotation_object(self.index)

        # Eliminar todas las capas de formas
        self.remove_all_shape_layers()

        # Crear las nuevas capas de formas
        self.shape_layers = self.create_shape_layers(self.annotation_object)

        # Reiniciar la vista
        self.viewer.reset_view()

    def get_image_id(self, index):
        return self.image_ids[index]

    def get_annotation_object(self, index):
        return self.annotations[index]

    def transform_x(self, x):
        # Re-normalizar
        x = re_normalize(x)

        return x

    def load_x(self, image_id):
        return imread(image_id)

    def load_annotations(self):
        # Generar una lista de nombres, el archivo de anotación debe tener el mismo nombre que la imagen.
        annotation_object_names = [
            annotation_object.name for annotation_object in self.annotations
        ]
        # Iterar sobre el identificadores de las anotaciones
        for annotation_id in self.annotation_ids:
            annotation_name = annotation_id.stem

            index_list = self.get_indices_of_sequence(
                annotation_name, annotation_object_names
            )
            if index_list:
                # Verificar si se encuentra mas de un índice
                idx = index_list[0]  # Obtener el valor de ínidce de index_list
                annotation_file = read_json(annotation_id)  # Leer archivo

                # Almacenarlos como np.ndarrays
                boxes = np.array(annotation_file["boxes"])  # Obtener las cajas
                boxes = np.array(
                    [make_bbox_napari(box) for box in boxes]
                )  # Transformar a cajas de napari
                labels = np.array(annotation_file["labels"])  # Obtener etiquetas

                # Añadir información a un objeto de anotación
                self.annotations[idx].boxes = boxes
                self.annotations[idx].labels = labels

    def get_indices_of_sequence(self, string, sequence):
        return [idx for idx, element in enumerate(sequence) if element == string]

    def get_annotations_from_shape_layers(self):
        all_shape_layers = self.get_all_shape_layers()
        if all_shape_layers:
            all_boxes = []
            all_labels = []
            for shape_layer in all_shape_layers:
                boxes = np.array(shape_layer.data)  # numpy.ndarray
                num_labels = len(boxes)
                label = shape_layer.metadata[
                    "label"
                ]  # Leer las etiquetas de los metadatosread the label from the metadata
                all_boxes.append(boxes)
                all_labels.append(np.repeat(np.array([label]), num_labels))

            all_boxes = np.concatenate(all_boxes, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            return all_boxes, all_labels

    def save_annotations(self, annotation_object):
        # Obtener la información de la anotación de cada capa de formas
        information = self.get_annotations_from_shape_layers()

        if information:
            boxes, labels = information  # Desempaquetamiento de tupla

            # Actualizar el objeto de anotación actual
            self.update_annotation_object(annotation_object, boxes, labels)

    def update_annotation_object(self, annotation_object, boxes, labels):
        annotation_object.boxes = boxes
        annotation_object.labels = labels

    def create_shape_layers(self, annotation_object):
        unique_labels = np.unique(annotation_object.labels)

        shape_layers = [
            self.create_shape_layer(label, annotation_object) for label in unique_labels
        ]

        return shape_layers

    def create_shape_layer(self, label, annotation_object):
        mask = annotation_object.labels == label

        boxes = annotation_object.boxes[mask]

        layer = self.viewer.add_shapes(
            data=boxes,
            edge_color=self.color_mapping.get(label, "black"),
            edge_width=self.edge_width,
            face_color="transparent",
            name=str(label),
        )

        layer.metadata["label"] = label

        return layer

    def add_class(self, label, color: str):
        self.color_mapping[label] = color
        layer = self.viewer.add_shapes(
            edge_color=self.color_mapping.get(label, "black"),
            edge_width=self.edge_width,
            face_color="transparent",
            name=str(label),
        )

        layer.metadata["label"] = label

    def export(self, directory: pathlib.Path, name: str = None):
        """Guardas las anotaciones actuales en disco."""
        self.save_annotations(
            self.annotation_object
        )  # Guardar las anotaciones en el annotation_object actual

        boxes = [
            make_bbox_napari(box, reverse=True).tolist()
            for box in self.annotation_object.boxes
        ]
        labels = self.annotation_object.labels.tolist()
        if name is None:
            name = pathlib.Path(self.annotation_object.name).with_suffix(".json")

        file = {"labels": labels, "boxes": boxes}

        save_json(file, path=directory / name)

        # with open(directory / name, 'w') as fp:
        #     json.dump(obj=file, fp=fp, indent=4, sort_keys=False)

        print(f"Annotation {str(name)} saved to {directory}")

    def export_all(self, directory: pathlib.Path):
        """Saves all available annotations to disk."""
        self.save_annotations(
            self.annotation_object
        )  # Guardar las anotaciones en el annotation_object actual

        for annotation_object in self.annotations:
            if annotation_object:
                boxes = [
                    make_bbox_napari(box, reverse=True).tolist()
                    for box in annotation_object.boxes
                ]
                labels = annotation_object.labels.tolist()
                name = pathlib.Path(annotation_object.name).with_suffix(".json")

                file = {"labels": labels, "boxes": boxes}

                save_json(file, path=directory / name)

                print(f"Annotation {str(name)} saved to {directory}")


class AnchorViewer(ViewerBase):
    def __init__(
        self,
        image: torch.tensor,
        rcnn_transform: GeneralizedRCNNTransform,
        feature_map_size: tuple,
        anchor_size: Tuple[tuple] = ((128, 256, 512),),
        aspect_ratios: Tuple[tuple] = ((1.0,),),
    ):
        self.image = image
        self.rcnn_transform = rcnn_transform
        self.feature_map_size = feature_map_size
        self.anchor_size = anchor_size
        self.aspect_ratios = aspect_ratios

        self.anchor_boxes = None

        # Instancia del visualizador napari
        self.viewer = None

    def napari(self):
        # IPython magic
        enable_gui_qt()

        # napari
        if self.viewer:
            try:
                del self.viewer
            except AttributeError:
                pass

        # Inicializar una instancia de napari
        self.viewer = napari.Viewer()

        # Mostrar imagen
        self.show_sample()

    def get_anchors(self):
        return get_anchor_boxes(
            self.image,
            self.rcnn_transform,
            self.feature_map_size,
            self.anchor_size,
            self.aspect_ratios,
        )

    def get_first_anchor(self):
        num_anchor_boxes_per_location = len(self.anchor_size[0]) * len(
            self.aspect_ratios[0]
        )
        return [self.anchor_boxes[idx] for idx in range(num_anchor_boxes_per_location)]

    def get_center_points(self):
        return get_center_bounding_box(self.anchor_boxes)

    def show_sample(self):
        self.anchor_boxes = self.get_anchors()
        self.first_anchor = self.get_first_anchor()
        self.center_points = self.get_center_points()
        self.anchor_points = self.center_points.unique(dim=0)

        # Transformar la imagen a numpy, CPU y correcto formato para visualizar
        image = self.transform_image(self.image)
        boxes = self.transform_boxes(self.first_anchor)

        # Crear una capa de imagen
        self.viewer.add_image(image, name="Image")

        # Crear una capa de formas
        self.viewer.add_shapes(
            data=boxes,
            face_color="transparent",
            edge_color="red",
            edge_width=2,
            name="Boxes",
        )

        # Crear una capa de puntos
        self.viewer.add_points(data=self.anchor_points)

        # Reiniciar vista
        self.viewer.reset_view()

    def transform_image(self, x):
        image_transformed = self.rcnn_transform([self.image])
        x = image_transformed[0].tensors[0]

        # Asrgurarse de que es un numpy.ndarray en el CPU
        x = x.cpu().numpy()

        # De [C, H, W] a [H, W, C] - únicamente imagenes en RGB.
        x = np.moveaxis(x, source=0, destination=-1)

        # Re-normalizar
        x = re_normalize(x)

        return x

    def transform_boxes(self, boxes):
        return [make_bbox_napari(box) for box in boxes]
