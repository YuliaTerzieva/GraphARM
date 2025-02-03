from torch_geometric.datasets import ZINC
import torch_geometric as pyg
from networkx import get_node_attributes
from tqdm import tqdm
import torch
from torch import nn
import math
import wandb
import os
import pickle as pkl

from models import DiffusionOrderingNetwork, DenoisingNetwork
from utils import NodeMasking
from grapharm import GraphARM
from torch_geometric.loader import DataLoader

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# instanciate the dataset
# dataset = ZINC(root='./data/ZINC', transform=None, pre_transform=None)


### For my understanding of the dataset here is what we are working with:
# print(type(dataset)) # <class 'torch_geometric.datasets.zinc.ZINC'>
# print(dataset.x[:10]) # tensor([[0],[0],[0],[0],[0],[0],[0],[2],[2],[0]])
# print(dataset.num_nodes)
# print(dataset.num_edges)
# print(dataset.num_node_features) # 1
# print(dataset.num_classes) # 218362
# print(dataset.is_undirected())

# print(type(dataset[0])) # <class 'torch_geometric.data.data.Data'>
# print(dataset[0]) # Data(x=[33, 1], edge_index=[2, 72], edge_attr=[72], y=[1])
# print(dataset[0].is_undirected()) # True

# print(dataset[0].num_nodes) # 33
# print(dataset[0].x)
"""tensor([[0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [2],
        [2],
        [0],
        [2],
        [0],
        [0],
        [0],
        [0],
        [0],
        [2],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1],
        [2],
        [0],
        [5]])"""

# print(dataset[0].num_edges) # 72
# print(dataset[0].edge_index)
""" tensor([[ 0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  6,  7,  7,  8,  8,
          8,  9,  9,  9, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 14, 15, 16,
         16, 16, 17, 17, 17, 18, 18, 19, 19, 20, 20, 20, 21, 22, 22, 23, 23, 23,
         24, 25, 25, 25, 26, 27, 27, 28, 28, 28, 29, 30, 30, 31, 31, 31, 32, 32],
        [ 1,  0,  2,  1,  3,  2,  4,  3,  5,  4,  6,  5,  7, 32,  6,  8,  7,  9,
         31,  8, 10, 11,  9,  9, 12, 28, 11, 13, 12, 14, 27, 13, 15, 16, 14, 14,
         17, 25, 16, 18, 23, 17, 19, 18, 20, 19, 21, 22, 20, 20, 23, 17, 22, 24,
         23, 16, 26, 27, 25, 13, 25, 11, 29, 30, 28, 28, 31,  8, 30, 32,  6, 31]])"""


nx_graphs = pkl.load(open(f'data/Ego_Nets_conf3', 'rb'))

# Create color mapping efficiently
colors = {color for g in nx_graphs for color in get_node_attributes(g, 'color').values()}
color_to_idx = {color: idx for idx, color in enumerate(colors)}


### I nede to turn the networkx data into torch_geometric data objects
pyg_graphs = []
for n in nx_graphs:
    """n.nodes[0] -> {'color': 'blue', 'anomaly': 0}"""
    for node, attrs in n.nodes(data=True):
        attrs['node_id'] = node  # Keep node ID
        attrs['color'] = color_to_idx.get(attrs.get('color', 'unknown'), -1)  # Convert color to int

    pyg_graphs.append(pyg.utils.from_networkx(n, group_node_attrs=['node_id', 'color']))

# Print first graph for verification
print(nx_graphs[0].nodes, nx_graphs[0].edges)
print(pyg_graphs[0].x, pyg_graphs[0].edge_index)

dataset = DataLoader(pyg_graphs, batch_size=32, shuffle=True)

diff_ord_net = DiffusionOrderingNetwork(node_feature_dim=2,
                                        num_node_types=dataset.x.unique().shape[0],
                                        num_edge_types=dataset.edge_attr.unique().shape[0],
                                        num_layers=3,
                                        out_channels=1,
                                        device=device)

masker = NodeMasking(dataset)


denoising_net = DenoisingNetwork(
    node_feature_dim=dataset.num_features,
    edge_feature_dim=dataset.num_edge_features,
    num_node_types=dataset.x.unique().shape[0],
    num_edge_types=dataset.edge_attr.unique().shape[0],
    num_layers=7,
    # hidden_dim=32,
    device=device
)


# wandb.init(
#         project="GraphARM",
#         group=f"v2.3.1",
#         name=f"ZINC_GraphARM",
#         config={
#             "policy": "train",
#             "n_epochs": 10000,
#             "batch_size": 1,
#             "lr": 1e-3,
#         },
#         # mode='disabled'
#     )

torch.autograd.set_detect_anomaly(True)


grapharm = GraphARM(
    dataset=dataset,
    denoising_network=denoising_net,
    diffusion_ordering_network=diff_ord_net,
    device=device
)

batch_size = 5
dataset = dataset[0:5]
try:
    grapharm.load_model()
    print("Loaded model")
except:
    print ("No model to load")
# train loop
for epoch in range(2):#2000
    print(f"Epoch {epoch}")
    grapharm.train_step(
        train_batch=dataset[2*epoch*batch_size:(2*epoch + 1)*batch_size],
        val_batch=dataset[(2*epoch + 1)*batch_size:batch_size*(2*epoch + 2)],
        M=4
    )
    grapharm.save_model()
