#!/usr/bin/env python


import pickle as pkl
import sys
import torch
import argparse

URLS = {
    'B_16': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth",
    'B_32': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32.pth",
    'L_32': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32.pth",
    'B_16_imagenet1k': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth",
    'B_32_imagenet1k': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32_imagenet1k.pth",
    'L_16_imagenet1k': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16_imagenet1k.pth",
    'L_32_imagenet1k': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32_imagenet1k.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(description='SimCLR converter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, help='path of the input tensorflow file')
    parser.add_argument('--out', type=str, help='path of the output pytorch path')
    args = parser.parse_args()
    return args


def vit2detectron(args):

    print("Converting from pytorch model to detectron model ...")
    obj = torch.load(args.input, map_location="cpu")

    if 'state_dict' in obj:
        obj = obj['state_dict']

    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "fc" in k and 'pwff' not in k:
            k = k.replace("fc", "fc1000")
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
    vit2detectron(args)
