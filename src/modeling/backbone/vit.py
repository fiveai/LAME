"""
    Taken from https://github.com/lukemelas/PyTorch-Pretrained-ViT#loading-pretrained-models
"""

from typing import Optional
import torch
from collections import defaultdict
from torch import nn
import logging

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from .transformer import Transformer
# from .utils import load_pretrained_weights
# from .configs import PRETRAINED_MODELS
from src.layers import (
    ShapeSpec,
)


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViT(Backbone):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000
    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self,
        config,
        load_repr_layer: bool = False,
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        image_size: Optional[int] = None,
    ):
        super().__init__()
        patches = config['patches']
        self.dim = config['dim']
        ff_dim = config['ff_dim']
        num_heads = config['num_heads']
        num_layers = config['num_layers']
        dropout_rate = config['dropout_rate']
        self.representation_size = config['representation_size']
        classifier = config['classifier']
        self.image_size = image_size                

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, self.dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.dim))
            seq_len += 1
        
        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, self.dim)
        else:
            raise NotImplementedError()
        
        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=self.dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)
        
        # Representation layer
        if self.representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(self.dim, self.representation_size)
            pre_logits_size = self.representation_size
        else:
            pre_logits_size = self.dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)

        # Initialize weights
        self.init_weights()
        
    def output_shape(self):
        if self.representation_size:
            return {'last': ShapeSpec(channels=self.representation_size)}
        else:
            return {'last': ShapeSpec(channels=self.dim)}

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        # nn.init.constant_(self.fc.weight, 0)
        # nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.
        Args:
            x (tensor): `b,c,fh,fw`
        """
        outputs = {}
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw+1,d 
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        x = self.norm(x)[:, 0]  # b,d
        outputs['last'] = x
        self.out = x.detach()
        return outputs

    def partition_parameters(self, granularity):

        logger = logging.getLogger(__name__)
        partition = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        section = 0
        logger.info(f"Partitionning ViT model with granularity {granularity}. {len(self.transformer.blocks)} blocks detected")
        for i, block in enumerate(self.transformer.blocks):
            for named_m, m in block.named_modules():
                immediate_parameters = list(m.named_parameters(recurse=False))
                for named_p, p in immediate_parameters:  # only get immediate parameters
                    name = f"block{i}_{named_m}.{named_p}"
                    if 'norm' in name:
                        type_ = 'BN'
                    elif 'weight' in name or 'bias' in name:
                        type_ = 'conv'
                    elif 'token' in name:
                        type_ = 'class_token'
                    elif 'embedding' in name:
                        type_ = 'positional_embedding'
                    else:
                        raise ValueError(f"Parameter {name} belongs to no category")
                    partition[section][type_]['names'].append(name)
                    partition[section][type_]['parameters'].append(p)  # we put the direct module parent as well
                    partition[section][type_]['modules'].append(m)  # we put the direct module parent as well
            if i % (len(self.transformer.blocks) // granularity) == 0 and i != 0:
                section += 1

        return self.default_to_regular(partition)


@BACKBONE_REGISTRY.register()
def build_vit_backbone(cfg, input_shape, **kwargs):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?

    config = PRETRAINED_MODELS[cfg.MODEL.VIT.NAME]['config']
    image_size = PRETRAINED_MODELS[cfg.MODEL.VIT.NAME]['image_size']
    return ViT(config=config, image_size=image_size)


def get_base_config():
    """Base ViT config ViT"""
    return dict(
          dim=768,
          ff_dim=3072,
          num_heads=12,
          num_layers=12,
          attention_dropout_rate=0.0,
          dropout_rate=0.1,
          representation_size=768,
          classifier='token'
    )


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(patches=(16, 16)))
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(patches=(32, 32)))
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        patches=(16, 16),
        dim=1024,
        ff_dim=4096,
        num_heads=16,
        num_layers=24,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representation_size=1024
    ))
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(patches=(32, 32)))
    return config


def drop_head_variant(config):
    config.update(dict(representation_size=None))
    return config


PRETRAINED_MODELS = {
    'B_16': {
      'config': get_b16_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth"
    },
    'B_32': {
      'config': get_b32_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32.pth"
    },
    'L_16': {
      'config': get_l16_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': None
    },
    'L_32': {
      'config': get_l32_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32.pth"
    },
    'B_16_imagenet1k': {
      'config': drop_head_variant(get_b16_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth"
    },
    'B_32_imagenet1k': {
      'config': drop_head_variant(get_b32_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32_imagenet1k.pth"
    },
    'L_16_imagenet1k': {
      'config': drop_head_variant(get_l16_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16_imagenet1k.pth"
    },
    'L_32_imagenet1k': {
      'config': drop_head_variant(get_l32_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32_imagenet1k.pth"
    },
}