from typing import List, Callable
import sys
import networkx as nx
import pickle
from tqdm import tqdm
from copy import deepcopy
# nltk.download('wordnet')
# nltk.download('omw')
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
import os 
from ..classes.imagenet import ID2NAME

from functools import partial
from multiprocessing import Pool


def fill_graph_from_file(G, filename):
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split(' ')
            src, dest = line
            G.add_edge(src, dest)


def pad_0_left(s: str, n: int):
    assert n >= len(s)
    padding = ''.join(["0"] * (n - len(s)))
    return padding + s


fns = {"synset2id": lambda x: f"{x.pos()}{pad_0_left(str(x.offset()), 8)}",
       "id2synset": lambda x: wn.synset_from_pos_and_offset(x[0], int(x[2:]) if x[1] == '0' else int(x[1:])),
       "synset2name": lambda x: x.name().split('.')[0],
       "nltkid2name": lambda x: x.split('.')[0],
       "id2name": lambda x: ID2NAME[x],
       "name2synsets": lambda x: wn.synsets(x),
       "process_name": lambda x: x.lower().replace(' ', '_'),
       }


def name2synset(name: str):
    """ Necessary for some retarded lists (e.g first 'whale') """
    synset_list = fns["name2synsets"](fns["process_name"](name))
    if name == "tiger":
        return synset_list[1]
    for synset in synset_list:
        if fns["process_name"](name) in synset.name():
            return synset
        else:
            print(name, synset.name())
    return synset


fns["name2synset"] = name2synset


def parallel_fill_graph(G, synset_list: List[Synset]):
    for i in tqdm(range(len(synset_list))):
        s_i = synset_list[i]
        partial_build = partial(build_paths, target=s_i, synset_transformer=fns["synset2id"])
        remaining_synsets = synset_list[i + 1:]
        with Pool(processes=4) as pool:
            all_paths = list(pool.map(partial_build, remaining_synsets))
        # for paths in mmap_(partial_build, remaining_synsets):
            # pass
            # print(paths)
        for paths in all_paths:
            for path in paths:
                for k in range(len(path) - 1):
                    n, np = path[k][0], path[k + 1][0]
                    if path[k][1] == 'down':
                        G.add_edge(n, np)
                    else:
                        G.add_edge(np, n)
        # return G


def sequential_fill_graph(G, synset_list: List[Synset]):
    for i in tqdm(range(len(synset_list))):
        for j in tqdm(range(i + 1, len(synset_list))):
            s_i = synset_list[i]
            s_j = synset_list[j]
            paths = build_paths(s_i, s_j, synset_transformer=fns["synset2id"])
            for path in paths:
                for k in range(len(path) - 1):
                    n, np = path[k][0].split('.')[0], path[k + 1][0].split('.')[0]
                    if path[k][1] == 'down':
                        G.add_edge(n, np)
                    else:
                        G.add_edge(np, n)
    # return G


def build_paths(source: Synset, target: Synset, synset_transformer: Callable):
    assert isinstance(source, Synset), isinstance(target, Synset)
    all_paths: List[List[Synset]] = []
    def dfs(root, prev_nodes, except_node=None):
        # if already_visited[root]:
        #     return
        if root is None:
            return
        prev_nodes.append((synset_transformer(root), "down"))
        if root == target:
            all_paths.append(deepcopy(prev_nodes))
        else:
            for child in root.hyponyms():
                if except_node is None or synset_transformer(child) != except_node:
                    dfs(child, prev_nodes)
        prev_nodes.pop(-1)
        # already_visited[root] = True
        return

    q = [([], source)] 
    while len(q):
        # print(q)
        prev_nodes, current_node = q.pop(0)
        dfs(current_node, prev_nodes, prev_nodes[-1][0] if len(prev_nodes) else None)
        updated_list = prev_nodes + [(synset_transformer(current_node), "up")]
        for parent in current_node.hypernyms():
            q.append((updated_list, parent))
    return all_paths


def get_imagenet_synsets() -> List[Synset]:
    synset_list = []
    for id_ in ID2NAME.keys():
        synset_list.append(fns["id2synset"](id_))
    return synset_list


if __name__ == "__main__":

    # Generate a sample DAG
    imagenet_synsets = get_imagenet_synsets()
    G = nx.DiGraph()
    if sys.argv[1] == 'plot':  # Plotting a sample of graph
        import pygraphviz as pgv
        G = pgv.AGraph(strict=True, directed=True)
        sequential_fill_graph(G, imagenet_synsets[:3])
        G.layout()
        G.draw(sys.argv[2])
        sys.exit()
    elif sys.argv[1] == 'build_parallel':  # Building and saving full graph
        parallel_fill_graph(G, imagenet_synsets)

    elif sys.argv[1] == 'build_sequential':  # Building and saving full graph
        sequential_fill_graph(G, imagenet_synsets)

    elif sys.argv[1] == 'build_from_file': 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(dir_path, 'wordnet.is_a.txt')
        fill_graph_from_file(G, filename)
    else:
        sys.exit()
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(G, f)
        print(f"Graph saved at {sys.argv[2]}")
