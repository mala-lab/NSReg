import torch
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import remove_self_loops
import numpy as np

def get_edge_mask(edge_index, signed_labels):
    assert isinstance(signed_labels, torch.Tensor)
    src_signs, tar_signs = signed_labels[edge_index[0]], signed_labels[edge_index[1]]

    # Choose pos
    pos_src_mask, pos_tar_mask = src_signs == 0, tar_signs == 0
    pos_mask = pos_src_mask & pos_tar_mask

    # Choose neg
    neg_src_mask, neg_tar_mask = src_signs == 1, tar_signs == 1
    neg_mask = neg_src_mask | neg_tar_mask

    return pos_mask, neg_mask


def subgraph_or(subset, edge_index, num_nodes):
    device = edge_index.device

    if isinstance(subset, (list, tuple, np.ndarray)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        num_nodes = subset.size(0)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        subset = index_to_mask(subset, size=num_nodes)

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] | node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]

    return edge_index

def subgraph_xor(subset, edge_index, num_nodes):
    device = edge_index.device

    if isinstance(subset, (list, tuple, np.ndarray)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        num_nodes = subset.size(0)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        subset = index_to_mask(subset, size=num_nodes)

    node_mask = subset
    edge_mask = torch.logical_xor(node_mask[edge_index[0]],  node_mask[edge_index[1]])
    edge_index = edge_index[:, edge_mask]


    return edge_index


def subgraph_and(subset, edge_index, num_nodes):
    device = edge_index.device

    if isinstance(subset, (list, tuple, np.ndarray)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        num_nodes = subset.size(0)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        subset = index_to_mask(subset, size=num_nodes)

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]

    return edge_index


def get_rand_idx(size, n):
    return torch.randperm(size)[:n]


# def remove_edges(edge_index, edges_to_remove):
#     edge_index = remove_self_loops(edge_index)[0]
#     edge_list = edge_index.t().tolist()

#     # Convert edges_to_remove to a list of tuples
#     edges_to_remove_list = edges_to_remove.t().tolist()

#     # Filter the edge_list to remove the specified edges
#     filtered_edge_list = [edge for edge in edge_list if edge not in edges_to_remove_list]

#     # Convert the filtered_edge_list back to edge_index tensor
#     filtered_edge_index = torch.tensor(filtered_edge_list).t()
#     return filtered_edge_index

def remove_edges(edge_index, edges_to_remove):
    edge_index = remove_self_loops(edge_index)[0]

    # Convert edge_index to a set of tuples
    edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Convert edges_to_remove to a set of tuples
    edges_to_remove_set = set(zip(edges_to_remove[0].tolist(), edges_to_remove[1].tolist()))

    # Filter the edge_set to remove the specified edges
    filtered_edge_set = edge_set - edges_to_remove_set

    # Convert the filtered_edge_set back to edge_index tensor
    filtered_edge_list = list(filtered_edge_set)
    filtered_edge_index = torch.tensor(filtered_edge_list, dtype=torch.long).t()
    return filtered_edge_index

def make_fully_connected(nodes):
    x, y = torch.meshgrid(nodes, nodes)

    # Reshape the coordinate tensors to 1-dimensional
    x = x.flatten()
    y = y.flatten()

    edges = torch.stack((x, y), dim=1)
    edges = edges.t()
    
    return edges

def test():
    edge_index = torch.LongTensor([[0, 1, 2, 1, 0], [1, 2, 3, 3, 3]])
    labels = torch.Tensor([1, 0, 0, 0])
    print(get_edge_mask(edge_index, labels))


if __name__ == "__main__":
    test()