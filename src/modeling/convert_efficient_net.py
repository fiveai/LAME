#!/usr/bin/env python


import pickle as pkl
import sys
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='EfficientNet converter',
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
        if "_fc" in k:
            k = k.replace("_fc", "fc1000")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}
    torch.save(res, args.out)
    print("Successfully converted and saved !!")
    if obj:
        print("Unconverted keys:", obj.keys())


if __name__ == "__main__":
    args = parse_args()
    pytorch2detectron(args)
