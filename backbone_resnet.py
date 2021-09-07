import torch
import torchvision.models as models
from torch import nn
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


def get_resnet_backbone(backbone_name: str):
    """
    Regresa una arquitectura base versión de ResNet pre-entrenada en ImageNet.
    Además remueve la capa de submuestreo promedio (average-pooling) y la capa
    lineal al final de la arquitectura.
    """
    if backbone_name == "resnet18":
        pretrained_model = models.resnet18(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == "resnet34":
        pretrained_model = models.resnet34(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == "resnet50":
        pretrained_model = models.resnet50(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == "resnet101":
        pretrained_model = models.resnet101(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == "resnet152":
        pretrained_model = models.resnet152(pretrained=True, progress=False)
        out_channels = 2048

    backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
    backbone.out_channels = out_channels

    return backbone


def get_resnet_fpn_backbone(
    backbone_name: str, pretrained: bool = True, trainable_layers: int = 5
):
    """
    Regresa una arquitectura base versión de ResNet con FPN
    pre-entrenada en ImageNet.
    """
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        trainable_layers=trainable_layers,
    )

    backbone.out_channels = 256
    return backbone


def resnet_fpn_backbone(
    backbone_name: str,
    pretrained: bool,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers: int = 3,
    returned_layers=None,
    extra_blocks=None,
):
    # Version adaptada del paquete original de PyTorch Vision.
    """
    Contruye una arquitectura base especificada versión de ResNet con FPN. Conjela un número
    específicado de capas en la arquitectura base.

    Argumentos:
        backbone_name (string): arquitectura ResNet. Los valores posibles son 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        norm_layer (torchvision.ops): Es recomendado utilizar el valor por defectp. Mas detalles en:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        pretrained (bool): Si es True, regresa un modelo con una arquitectura base pre-entrenada en Imagenet.
        trainable_layers (int): Numero de capas ResNet entrenables (no congeladas) comenzando a partir del bloque final.
            Valores validos entre 0 y 5, 5 que todas las capas de la arquitectura base son entrenables.
    """
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained, norm_layer=norm_layer
    )

    # Selección de las capas que no serán congeladas
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers
    ]
    # Conjelar las capas si una arquitectura base pre-entrenada es utilizada
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
    )


class BackboneWithFPN(nn.Module):
    """
    Añade una FPN en el tope del modelo.
    Internamente, usa el módulo torchvision.models._utils.IntermediateLayerGetter
    para extraer un submodelo que regresa un mapa de características especificado en
    return_layers. También aplica la misma limitación IntermediatLayerGetter.

    Argumentos:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): un diccionario que contiene los nombres
            de los módulos  para los cuales las activaciones serán retornadas como las
            llaves del diccionario, y el valor del diccionario es el nombre de la
            activación retornada (que el usuario puede especificar).
        in_channels_list (List[int]): Es el número de canales para cada mapa de características
            que es regresado, en el orden que son presentados en el OrderedDict.
        out_channels (int): número de canales en la FPN.

    Atributos:
        out_channels (int): número de canales en la FPN.
    """

    def __init__(
        self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None
    ):
        super(BackboneWithFPN, self).__init__()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
