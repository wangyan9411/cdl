# pylint: skip-file

import json, math
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from baidu_map import xBaiduMap


def load_location(file):
    """load location info including city and address, from a json format file """
    locations = dict()
    with open(file, 'r') as f:
        content = json.load(f)
        for note in content:
            # skip these items, whose location, city or address is missing
            if 'location' not in note.keys() or 'city' not in note['location'] or note['location']['city'] == ""\
                    or 'address' not in note['location'] or note['location']['address'] == "":
                continue
            else:
                locations[note['id']] = note['location']
    return locations


def transform_location(location, save_file):
    """transform the location from address to longitude and latitude, and write info into a file"""
    bm = xBaiduMap()
    newloc = dict()
    for id, loc in location.items():
        item = bm.getLocationByAddress(loc['address'].encode('utf-8'), loc['city'].encode('utf-8'))
        #skip these items, whose longitude and latitude cannot obtain from baidu map API
        if item is not None:
            newloc[id] = item
    with open(save_file, 'w') as f:
        f.write(json.dumps(newloc))


def judge_edge(loc1, loc2):
    """judge whether the edge form loc1 and loc2 should be add into the edges list, according to the distance
     between these two location and the user-defined threshold"""
    threshold = 5
    distance = (loc1[1] - loc2[1]) * (loc1[1] - loc2[1]) + (loc1[2] - loc2[2]) * (loc1[2] - loc2[2])
    if distance < threshold:
        return distance
    return 0


def construct_graph(file):
    """read location info from a file, and construct a graph from these info with networkx format"""
    locations = list()
    with open(file, 'r') as f:
        content = json.load(f)
        for key, item in content.items():
            item = [key, item[0], item[1]]
            locations.append(item)
    n = len(locations)
    graph = nx.Graph()
    for i in range(n):
        for j in range(n):
            weight = judge_edge(locations[i], locations[j])
            if i != j and weight != 0:
                graph.add_edge(i, j, weight=1. / math.sqrt(weight))
    return graph


def network_embedding(nx_G, output):
    """run network embedding with node2vec algorithm"""
    p = 1.0
    q = 1.0
    num_walks = 2
    walk_length = 100
    dimensions = 10
    window_size = 5
    workers = 5
    iter = 1
    G = node2vec.Graph(nx_G, False, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, iter=iter)
    model.save_word2vec_format(output)


if __name__ == '__main__':
    file_name = 'activity.txt'
    transform_file = 'location.json'
    embedding_file = 'embedding.dat'
    location = load_location(file_name)
    transform_location(location, transform_file)
    nx_G = construct_graph(transform_file)
    network_embedding(nx_G, embedding_file)


