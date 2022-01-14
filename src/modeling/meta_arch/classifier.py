# Copyright (c) Malik Boudiaf
from typing import Dict, Tuple
import torch
import logging
import numpy as np
from typing import List, Any
from torch import nn
from torch.nn import functional as F
from functools import partial
from src.config import configurable
from src.layers import Conv2d
from src.structures import ImageList, Instances
from src.data import DatasetCatalog

from ..backbone import Backbone, build_backbone
from .build import META_ARCH_REGISTRY
__all__ = ["Classifier"]


@META_ARCH_REGISTRY.register()
class Classifier(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        cls_in_features: List[str],
        **kwargs,
    ):
        """
        Args:
            backbone: a backbone module, must follow src's backbone interface
            CLS_HEAD: a module that predicts semantic segmentation from backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.super2sub_mapping = kwargs["super2sub"]
        self.cls_classes = kwargs['cls_classes']
        self.num_classes = kwargs['num_classes']
        self.normalize_input = kwargs['normalize_input']
        self.standardize_input = kwargs['standardize_input']
        self.head_type = kwargs['head_type']

        # -------- Classifier -----------
        input_shape = {k: v for k, v in backbone.output_shape().items() if k in cls_in_features}
        # input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        input_shape = input_shape.items()
        self.in_features = [k for k, v in input_shape]
        self.feature_dim = sum([v.channels for k, v in input_shape])
        if self.head_type == 'conv2d':
            self.fc1000: nn.Module = nn.Conv2d(self.feature_dim, self.cls_classes, kernel_size=1)  # conv2d instead of linear for checkpoint compatibility
        else:
            self.fc1000 = nn.Linear(self.feature_dim, self.cls_classes)  # conv2d instead of linear for checkpoint compatibility

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    def get_out_features(self, stage_index):
        if stage_index == 0:
            return self.backbone.stem.out
        else:
            return self.backbone.stages[stage_index - 1][-1].out

    @classmethod
    def from_config(cls, cfg):
        norm = nn.BatchNorm2d
        layers = {'conv': Conv2d,
                  'linear': nn.Linear,
                  'norm': partial(norm,
                                  momentum=cfg.MODEL.BACKBONE.BN_MOMENTUM,
                                  )
                  }
        backbone = build_backbone(cfg, layers=layers)
        data_names = getattr(cfg.DATASETS, 'ADAPTATION')[0]

        cls_classes = cfg.MODEL.CLS_HEAD.NUM_CLASSES

        num_classes = len(DatasetCatalog.get(data_names).thing_classes)

        if not DatasetCatalog.get(data_names).is_mapping_trivial:
            _, super2sub = DatasetCatalog.get(data_names).generate_mappings(cfg)
        else:
            super2sub = None

        return {
            "backbone": backbone,
            "cls_in_features": cfg.MODEL.CLS_HEAD.IN_FEATURES,
            "num_classes": num_classes,
            "cls_classes": cls_classes,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "super2sub": super2sub,
            "normalize_input": cfg.MODEL.NORMALIZE_INPUT,
            "standardize_input": cfg.MODEL.STANDARDIZE_INPUT,
            "head_type": cfg.MODEL.CLS_HEAD.TYPE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict[str, Any]]):
        probas = self.get_probas(batched_inputs)
        results = self.format_result(batched_inputs, probas)
        return results

    def normalize(self, batched_inputs: List[Dict[str, Any]], key: str):

        image_list = [torch.as_tensor(np.ascontiguousarray(x[key].transpose(2, 0, 1))) for x in batched_inputs]
        image_list = [x.to(self.device) for x in image_list]
        if self.normalize_input:
            image_list = [x.to(self.device) / 255. for x in image_list]
        if self.standardize_input:
            image_list = [(x - self.pixel_mean) / self.pixel_std for x in image_list]
        images = ImageList.from_tensors(image_list, self.backbone.size_divisibility, pad_value=0.)
        return images.tensor

    def extract_features(self, batched_inputs: List[Dict[str, Any]], key: str = "image"):

        if isinstance(batched_inputs, list):
            tensor = self.normalize(batched_inputs, key)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise ValueError("Input format not recognized.")

        features = self.backbone(tensor)

        gap_features = {}
        for i, f in enumerate(self.in_features):
            feature_map = features[f]  # [N, c, h, w]

            if len(feature_map.size()) == 4:  # Convolutional backbone
                gap_features[f] = self.avgpool(feature_map, valid_pixels=None)
            else:
                # ViT
                gap_features[f] = feature_map

            assert len(gap_features[f].size()) == 2
        cat_features = torch.cat(list(gap_features.values()), dim=1)  # [N, c, 1, 1]
        if 'conv' in self.head_type:
            cat_features = cat_features.view(cat_features.size(0), cat_features.size(1), 1, 1)
        return cat_features, gap_features

    def forward_head(self, features: torch.Tensor):
        """
        features: [N, c, 1, 1]
        weights: # [K, c, 1, 1]
        """

        logits = self.fc1000(features)  # [N, K, 1, 1] or [N, K]
        logits = logits.view(-1, logits.size(1))
        return logits

    def remap_probas(self, probas):
        
        if self.super2sub_mapping is not None:
            selected_probas = torch.zeros(probas.size(0), len(self.super2sub_mapping)).to(probas.device)
            for i in range(len(self.super2sub_mapping)):

                select_classes = self.super2sub_mapping[i]
                selected_probas[:, i] = probas[:, select_classes].sum(-1)

            # === Renormalize the probabilities ===
            selected_probas /= selected_probas.sum(-1, keepdim=True)
            probas = selected_probas
        return probas

    def get_probas(self, batched_inputs: List[Dict[str, Any]],
                   key: str = "image", remap=True):

        cat_features, gap_features = self.extract_features(batched_inputs, key)
        logits = self.forward_head(cat_features)
        probas = logits.softmax(-1)
        if remap:
            probas = self.remap_probas(probas)
        return probas

    def avgpool(self, x: torch.Tensor, valid_pixels: torch.Tensor) -> torch.Tensor:
        """
        x: feature maps of shape [N, C, h, w]
        """
        # valid_pixels = x.max(dim=1, keepdim=True) > 0
        if valid_pixels is not None:
            pooled_map = (x * valid_pixels).sum(dim=(-1, -2)) / valid_pixels.sum(dim=(-1, -2))
        else:
            pooled_map = x.mean((-2, -1))
        return pooled_map

    def format_result(self, batched_inputs: List[Dict[str, Any]],
                      probas: torch.tensor) -> Dict[str, Any]:

        image_sizes = [x["image"].shape for x in batched_inputs]
        results = {'probas': probas}
        gt = torch.tensor([obj["instances"].gt_classes[0] for obj in batched_inputs]).to(self.device)
        results['gts'] = gt
        results['one_hot_gts'] = F.one_hot(gt, self.num_classes)
        results["instances"] = []
        with torch.no_grad():
            for i, image_size in enumerate(image_sizes):
                res = Instances(image_size)
                # print(list(batched_inputs[i].keys()))
                gt_instances = batched_inputs[i]["instances"]
                res.gt_classes = gt_instances.gt_classes
                n_instances = len(gt_instances.gt_classes)
                res.pred_classes = probas[i].argmax(-1, keepdim=True).repeat(n_instances)
                res.scores = probas[i].max(-1, keepdim=True).values.repeat(n_instances)
                results["instances"].append(res)
        return results
