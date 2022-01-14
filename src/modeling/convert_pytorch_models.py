#!/usr/bin/env python


import pickle as pkl
import sys
import torch
import argparse

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

"""
Usage:
  # download one of the ResNet{18,34,50,101,152} models from torchvision:
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O r50.pth
  # run the conversion
  ./convert-torchvision-to-d2.py r50.pth r50.pkl
  # Then, use r50.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/r50.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
INPUT:
  FORMAT: "RGB"
  These models typically produce slightly worse results than the
  pre-trained ResNets we use in official configs, which are the
  original ResNet models released by MSRA.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='SimCLR converter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, help='path of the input tensorflow file')
    parser.add_argument('--out', type=str, help='path of the output pytorch path')
    args = parser.parse_args()
    return args


def pytorch2detectron(args):

    print("Converting from pytorch model to detectron model ...")
    obj = torch.load(args.input, map_location="cpu")

    if 'state_dict' in obj:
        obj = obj['state_dict']

    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "fc" in k:
            k = k.replace("fc", "fc1000")
        elif "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    # with open(args.out, "wb") as f:
    #     torch.save(res, f)
    torch.save(res, args.out)
    print("Successfully converted and saved !!")
    if obj:
        print("Unconverted keys:", obj.keys())


if __name__ == "__main__":
    args = parse_args()
    pytorch2detectron(args)
