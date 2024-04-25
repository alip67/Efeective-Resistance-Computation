import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import wspd
import time


import torch
import torch_geometric
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform

import networkx as nx
from functools import reduce



from Network import *



def get_and_add_box(dumbell_indices,boxes,data, i):
    l,r = dumbell_indices[i]
    # loop loop loop loop
    lbox = np.array([[min(x),max(x)] for x in data[l].T]).T
    rbox = np.array([[min(x),max(x)] for x in data[r].T]).T
    boxes.append([lbox,rbox])

def compute_centers(boxes, lcenters, rcenters, i):
    lc, rc = boxes[i]
    lcenters.append( (lc[0,:] + lc[1,:])/2)
    rcenters.append( (rc[0,:] + rc[1,:])/2)


def make_binary_tree_graph(r, h):
    T = nx.balanced_tree(r, h)
    # print("Graph Nodes:", list(T.nodes))
    # print("Graph Edges:", list(T.edges))
    TT = nx.adjacency_matrix(T)
    # print("Adjacency Matrix:", TT.todense())

    # pos = graphviz_layout(T, prog="dot")
    # nx.draw(T, pos)
    # plt.show()

    matrix = nx.linalg.graphmatrix.adjacency_matrix(T).todense()
    print(type(matrix))
    matrix = np.array(matrix).astype(float)
    return matrix, T


def make_2d_graph(m, n, periodic=False, return_pos=False):
    network = nx.grid_2d_graph(m, n, periodic=False, create_using=None)
    matrix = nx.linalg.graphmatrix.adjacency_matrix(network).todense()
    matrix = np.array(matrix).astype(float)

    pos = graphviz_layout(network, prog="dot")
    nx.draw(network, pos)
    plt.show()

    return matrix, network


def make_path_graph(n):
    parent = [i for i in range(n-1)]
    children = [i+1 for i in range(n-1)]
    external = [1, n]

    P = nx.path_graph(n)

    # print("Parent:", parent)
    # print("Children:", children)
    # print("External:", external)
    # print("Graph Nodes:", list(P.nodes))
    # print("Graph Edges:", list(P.edges))
    # Adj_P = nx.adjacency_matrix(P)
    # print("Adjacency Matrix:", Adj_P.todense())

    pos = graphviz_layout(P, prog="dot")
    nx.draw(P, pos)
    plt.show()


    matrix = nx.linalg.graphmatrix.adjacency_matrix(P).todense()
    Adj_P = np.array(matrix).astype(float)
    # return Adj_P, parent, children, n, external
    return Adj_P, P

def calc_ef_embedding(graph):
  laplacian = torch.tensor(nx.laplacian_matrix(graph).todense().astype("float"))
  pinv = torch.linalg.pinv(laplacian, hermitian=True)
  squared_pinv = pinv @ pinv


  # The entries resistance_matrix[s,t] are the effective resistance  between s and t.
  pinv_diagonal = torch.diagonal(pinv)
  resistance_matrix = pinv_diagonal.unsqueeze(0) + pinv_diagonal.unsqueeze(1) - 2*pinv
  return resistance_matrix



n= 10
adj_path, path_graph = make_path_graph(n)


ef_embedding_path = calc_ef_embedding(path_graph)
print(ef_embedding_path)


n = 4
m = 4
adj_grid, grid_graph = make_2d_graph(n,m)

network = Network(None, None, grid_graph)
epsilon=0.001
method='spl' #spl #ext
Effective_R = network.effR(epsilon, method)
print(Effective_R)


# S = 2.0 # separation constant

# # Don't provide multiple copies of the same point in the data. The copies will get anyways removed in build_wspd() call.
# data = np.array(Effective_R)
# data_pts = data.tolist()


# nr_pts = len(data_pts) # number of points
# print("number of points: " f'{nr_pts}')
# dim = len(data_pts[0]) # point dimension
# print("point dimension: " f'{dim}')

# data_pts = [wspd.point(p) for p in data_pts] # move points to point class objects

# tic = time.perf_counter()
# dumbells = wspd.build_wspd(nr_pts, dim, S, data_pts) # compute WSPD
# toc = time.perf_counter()

# print(f"WSPD construction in  {toc - tic:0.4f} seconds and size {len(dumbells)}")
# print(dumbells)