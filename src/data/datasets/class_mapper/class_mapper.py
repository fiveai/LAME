import numpy as np
from typing import List, Tuple, Dict, Any
import pprint
import logging
import networkx as nx
import pickle
import os
from collections import defaultdict
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from networkx.algorithms.dag import ancestors
import pandas as pd

from .build_imagenet_graph import fill_graph_from_file
from .build import MAPPER_REGISTRY

from ..classes.imagenet import \
        (ID2INDEX,
         ID2NAME,
         )

logger = logging.getLogger(__name__)


def pad_0_left(s: str, n: int):
    assert n >= len(s)
    padding = ''.join(["0"] * (n - len(s)))
    return padding + s


fns = {"synset2id": lambda x: f"{x.pos()}{pad_0_left(str(x.offset()), 8)}",
       "id2synset": lambda x: wn.synset_from_pos_and_offset(x[0], int(x[2:]) if x[1] == '0' else int(x[1:])),
       "synset2name": lambda x: x.name().split('.')[0],
       "id2name": lambda x: ID2NAME[x] if x in ID2NAME else None,
       "name2synsets": lambda x: wn.synsets(x),
       "process_name": lambda x: x.lower().replace(' ', '_'),
       }


def name2synset(name: str):
    """ Necessary for some retarded lists (e.g first 'whale') """
    synset_list = fns["name2synsets"](fns["process_name"](name))
    assert len(synset_list), f"Class {name} does not correspond to any known Synset"
    if name in ["tiger", "turtle"]:
        return synset_list[1]
    for synset in synset_list:
        if fns["process_name"](name) in synset.name():
            return synset
        else:
            print(name, synset.name())
    return synset


fns["name2synset"] = name2synset


def get_imagenet_synsets() -> List[Synset]:
    synset_list = []
    for id_ in ID2NAME.keys():
        synset_list.append(fns["id2synset"](id_))
    return synset_list


class ClassMapper:

    def __init__(self, cfg):
        self.threshold = cfg.DATASETS.MAPPER.THRESHOLD
        self.synset_similarity_fn = cfg.DATASETS.MAPPER.SYNSET_SIM
        self.output_dir = cfg.OUTPUT_DIR

    def get_mappings(self,
                     target_class_names: List[str],
                     target_class_ids: List[str],
                     target_synsets: List[str],
                     ):
        """ 
        returns:
            - imagenet2target maps ImageNet ids (e.g {"n0234982": })
        """
        logger.info("Obtaining mappings")

        # ------- Get mapping from imagenet_ids to target_names -------

        imagenet2target = self.get_mapping(target_class_names, target_synsets)
        assert len(imagenet2target) <= len(ID2NAME)

        # ------- Check classes that have been properly (mask variable) -------

        target2imagenet = defaultdict(list)
        for k, v in imagenet2target.items(): target2imagenet[v].append(fns['id2name'](k))
        pretty_dict_str = pprint.pformat(target2imagenet)
        logger.info(pretty_dict_str)
        mask = [1] * len(target_class_names)
        res: Dict[str, Any] = {}
        if len(target2imagenet) != len(target_class_names):
            mask = [x in target2imagenet for x in target_class_names]
            logger.warning("{} superclasses {} have not been mapped !".format(
                len(target_class_names) - len(target2imagenet),
                [x for x in target_class_names if x not in target2imagenet]),
            )
            logger.warning(f"Now remaining {len(target2imagenet)} superclasses")

        # ------- Recompute new classes if they have not been mapped -------
        res["new_names"] = [x for i, x in enumerate(target_class_names) if mask[i]]
        res["new_ids"] = [x for i, x in enumerate(target_class_ids) if mask[i]]
        if target_synsets is not None:
            res["new_synsets"] = [x for i, x in enumerate(target_synsets) if mask[i]]
        else:
            res["new_synsets"] = None

        # Prepare imagenet2target mapping
        res["imagenet2target"] = {}
        for imagenet_id, tgt_name in imagenet2target.items():
            index = res["new_names"].index(tgt_name)
            res["imagenet2target"][imagenet_id] = {"super_id": res["new_ids"][index],
                                                   "super_name": res["new_names"][index],
                                                   "super_index": index}
        res["target2imagenet"] = target2imagenet

        return res

    def save_mapping(self, mapping: Dict[Any, Any], class_count: Dict[str, int]):

        data = [(k, "; ".join(v), class_count[k]) for k, v in mapping.items()]
        df = pd.DataFrame.from_records(data, columns=['Target class', 'ImageNet classes', 'Number of remaining '])
        mapper_name = self.__class__.__name__
        with pd.option_context("max_colwidth", 1000):
            df.to_latex(os.path.join(self.output_dir, f'{mapper_name}.tex'),
                        caption=f"Mapping obtained with mapper {mapper_name}.",
                        label=f"tab:{mapper_name}",
                        index=False)
        df.to_csv(os.path.join(self.output_dir, f'{mapper_name}.csv'), index=False)

    def get_mapping(self, target_class_names: List[str], target_synsets: List[str]):
        raise NotImplementedError


@MAPPER_REGISTRY.register()
class ThresholdSynsetMapper(ClassMapper):

    def get_mapping(self, target_class_names: List[str], target_synsets: List[str]):
        """
        Maps all synsets from {a: b}, where a is in synsets_a and b in synsets_b. This is done by computing similarity.
        """
        # ------ Retrieve synsets ------------
        imagenet_synsets = get_imagenet_synsets()
        if target_synsets is None:
            target_synsets = list(map(lambda x: name2synset(x), target_class_names))
        else:
            target_synsets = list(map(lambda x: wn.synset(x), target_synsets))
        assert len(target_synsets) == len(target_class_names),\
            "Some classes are not synsets"

        # ------ Get mapping class_imagenet: target_class ------------
        sim_tuples = list(map(lambda x: self.most_similar(target_synsets, x), imagenet_synsets))
        filtered_tuples = list(filter(lambda t: t[2] > self.threshold, sim_tuples))
        mapping = {fns["synset2id"](t[0]): target_class_names[t[1]] for t in filtered_tuples}
        return mapping

    def most_similar(self, gallery: List[Synset], target_synset: Synset):
        # print(str(gallery) + '------------------------------')
        sim_fn = eval(f'wn.{self.synset_similarity_fn}')
        similarity_list = np.array(list(map(lambda x: sim_fn(target_synset, x), gallery)))
        # if 'tiger' in fns["synset2name"](target_synset):
        #     similarity = [(fns["synset2name"](x), s) for x, s in zip(gallery, similarity_list)]
        #     print("{} : {} \n".format(
        #         fns["synset2name"](target_synset), similarity))
        top_match = np.argmax(similarity_list)
        return (target_synset, top_match, similarity_list[top_match])


@MAPPER_REGISTRY.register()
class AncestralSynsetMapper(ClassMapper):

    def __init__(self, cfg):
        super(AncestralSynsetMapper, self).__init__(cfg)
        path = cfg.DATASETS.MAPPER.IMAGENET_GRAPH_PATH
        if os.path.exists(path):
            with open(path, 'rb') as f:
                graph = pickle.load(f)
        else:
            graph = nx.DiGraph()
            dir_path = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(dir_path, 'wordnet.is_a.txt')
            fill_graph_from_file(graph, filename)
            logger.info(f"ImageNet graph not detected. Building ImageNet graph using  {filename}...")
            with open(path, 'wb') as f:
                pickle.dump(graph, f)
                logger.info(f"Saved graph at {path}...")
        self.graph = graph

    def map_to_closest_synset(self, synset: Synset, gallery: List[Synset]):
        # synset = fns["name2synset"](class_name)
        id_ = fns["synset2id"](synset)
        if id_ in self.graph:
            return id_
        else:
            logger.warning(f"Synset {id_} was not in the graph nodes")
            sim_fn = eval(f'wn.{self.synset_similarity_fn}')
            similarities = list(map(lambda x: sim_fn(synset, x), gallery))
            top_match = np.argmax(similarities)
            matching_id = fns["synset2id"](gallery[top_match])
            assert matching_id in self.graph
            return matching_id

    def get_mapping(self, target_class_names: List[str], target_synsets: List[str], out_format='id2id') -> Dict[Any, Any]:
        self.all_ids = self.get_all_ids()

        # ------ Retrieve synsets ------------

        imagenet_ids = list(ID2NAME.keys())
        if target_synsets is None:
            target_synsets = list(map(lambda x: name2synset(x), target_class_names))
        else:
            if len(target_synsets[0].split('.')) > 1:
                target_synsets = list(map(lambda x: wn.synset(x), target_synsets))
            else:
                target_synsets = list(map(lambda x: fns["id2synset"](x), target_synsets))

        # ------ Need to map target_synsets to the closest synset in the graph -------

        synset_gallery: List[Synset] = list(map(lambda x: fns["id2synset"](x), self.graph.nodes()))
        closest_ids: List[str] = list(map(lambda s: self.map_to_closest_synset(s, synset_gallery), target_synsets))

        # ------ Retrieve parents for each imagenet synset -----------

        imagenet_parents: List[Tuple[str, List[str]]] = list(map(lambda x: self.find_ancestors(x, closest_ids), imagenet_ids))

        # ------ Filter out synsets with no parents -----------

        filtered_imagenet_parents: List[Tuple[str, List[str]]] = list(filter(lambda x: len(x[1]) > 0, imagenet_parents))

        # ------ Only keep a single parent with highest similiarity -----------

        def get_best_parent(id_: str, parent_ids: List[str]) -> Tuple[str, str]:
            sim_fn = eval(f'wn.{self.synset_similarity_fn}')
            parent_synsets = list(map(fns["id2synset"], parent_ids))
            synset = fns["id2synset"](id_)
            similarities = list(map(lambda x: sim_fn(x, synset), parent_synsets))
            top_match = np.argmax(similarities)
            if len(parent_ids) > 1:
                parent_synsets = [fns["id2synset"](x) for x in parent_ids]
                logger.warning(f"Conflict for class {ID2NAME[id_]}. Chose {parent_synsets[top_match]} from {parent_synsets}")
            return id_, parent_ids[top_match]
        child_parent: List[Tuple[str, str]] = list(map(lambda t: get_best_parent(t[0], t[1]), filtered_imagenet_parents))

        # ------ Put in expected output format -----------
        if out_format == 'id2id':
            mapping = {t[0]: target_class_names[closest_ids.index(t[1])] for t in child_parent}
        elif out_format == 'index2index':
            mapping = {ID2INDEX[t[0]]: closest_ids.index(t[1]) for t in child_parent}  # type: ignore[misc]
        return mapping

    def find_ancestors(self, id_: str, id_gallery: List[str]) -> Tuple[str, List[str]]:

        id_ancestors = ancestors(self.graph, id_).union({id_})
        return id_, list(id_ancestors.intersection(set(id_gallery)))

    def get_all_ids(self):
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(dir_path, 'words.txt')        
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        lines = [x.split('\t') for x in lines]
        dic = {x[0]: x[1].split('/') for x in lines}
        return dic
