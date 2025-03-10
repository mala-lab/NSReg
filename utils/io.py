import numpy as np
import yaml
import scipy.sparse as sp
import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.utils import is_undirected, to_undirected, dense_to_sparse
from torch_geometric.data import Data
from utils.data import DglDataset


def load_npz(fp):
    data = np.load(fp, allow_pickle=True)

    return data


def load_dgl_graph(dset_name, homo=1, view=None):
    graph = DglDataset(dset_name, homo=homo, view=view).graph
    x_all, adj = graph.ndata['feature'], graph.adj().to_dense()
    return x_all, adj


def make_pyg_graph_dgl(x, adj, undirected=True):
    edge_index = dense_to_sparse(adj)[0]
    if undirected:
        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)

    data = Data(x=x, edge_index=edge_index)

    if undirected:
        assert data.is_undirected()

    return data


def make_pyg_graph(x, adj, undirected=True):
    features = torch.from_numpy(x.todense()).float()
    edge_index, _ = from_scipy_sparse_matrix(adj)

    if undirected:
        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)

    data = Data(x=features, edge_index=edge_index)

    if undirected:
        assert data.is_undirected()

    return data


def dgl_data_to_pyg_graph(x_all, adj, class_labels):
    pyg_graph = make_pyg_graph_dgl(x_all, adj, undirected=True)

    class_idx, class_size = torch.unique(class_labels, return_counts=True)
    class_per = class_size / class_labels.size(0)
    class_names = ["n_%2d" % i for i in range(class_idx.size(0))] + ["a_%2d" % i for i in range(class_idx.size(0))]
    print(class_per)

    dset_info = {
        'class_idx': class_idx,
        'class_size': class_size,
        'class_per': class_per,
        'class_names': class_names
    }

    return pyg_graph, dset_info


# data is the return of the load_npz function
# Heavily modified https://github.com/shchur/gnn-benchmark/blob/master/gnnbench/data/io.py
def npz_data_to_pyg_graph(data):
    # Load adj matrix
    adj_matrix = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), shape=data['adj_shape'])

    # Load feature matrix
    if 'attr_data' in data:
        # Attributes are stored as a sparse CSR matrix
        attr_matrix = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']),
                                    shape=data['attr_shape'])
    elif 'attr_matrix' in data:
        # Attributes are stored as a (dense) np.ndarray
        attr_matrix = data['attr_matrix']
    else:
        attr_matrix = None

    # Load label matrix
    if 'labels_data' in data:
        # Labels are stored as a CSR matrix
        labels = sp.csr_matrix((data['labels_data'], data['labels_indices'], data['labels_indptr']),
                               shape=data['labels_shape'])
    elif 'labels' in data:
        # Labels are stored as a numpy array
        labels = data['labels']
    else:
        labels = None

    class_idx, class_size = np.unique(labels, return_counts=True)
    class_per = class_size/labels.shape[0]

    print(class_per)

    graph = make_pyg_graph(attr_matrix, adj_matrix, undirected=True)
    dset_info = {
        'node_names': data.get('node_names'),
        'attr_names': data.get('attr_names'),
        'class_names': data.get('class_names'),
        'metadata': data.get('metadata'),
        'class_idx': class_idx,
        'class_size': class_size,
        'class_per': class_per,
    }

    return graph, labels, dset_info

def load_yaml(fn):
    with open(fn) as fp:
        config = yaml.safe_load(fp)
    return config