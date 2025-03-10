import numpy as np
import torch
from dgl.data import FraudYelpDataset
import dgl

def random_split(idx, train_ratio):
    n_train = int(idx.shape[0] * train_ratio)
    randperm = torch.randperm(idx.shape[0])
    return idx[randperm[:n_train]], idx[randperm[n_train:]]


def num_split(idx, n_train):
    randperm = torch.randperm(idx.shape[0])
    return idx[randperm[:n_train]], idx[randperm[n_train:]]


def select_class_idx_in_list(labels, classes):
    node_idx = None
    for i in classes:
        cur_idx = np.where(labels == i)[0].flatten()
        node_idx = cur_idx if node_idx is None else np.hstack((node_idx, cur_idx))
    return node_idx

def get_classes_idx(all_idx, labels, classes):
    selected_idx = None
    for i in classes:
        idx = all_idx[np.where(labels==i)]
        selected_idx = idx if selected_idx is None else np.hstack((selected_idx, idx))
    return selected_idx


def get_split_idx(node_idx, labels, all_classes, known_anomaly_classes, unknown_anomaly_classes):
    normal_classes = [i for i in all_classes if i not in known_anomaly_classes and i not in unknown_anomaly_classes]

    return {
        'normal_idx': get_classes_idx(node_idx, labels, normal_classes),
        'known_idx': get_classes_idx(node_idx, labels, known_anomaly_classes),
        'unknown_idx': get_classes_idx(node_idx, labels, unknown_anomaly_classes),
    }


def ad_split(labels, train_ratio, class_info):
    known_anomaly = class_info['known_anomaly']
    unknown_anomaly_classes = class_info['unknown_anomaly']
    normal_classes = class_info['normal']



    known_anomaly_idx = np.where(labels==known_anomaly)[0].flatten()
    normal_idx = select_class_idx_in_list(labels, normal_classes)
    unknown_anomaly_idx = select_class_idx_in_list(labels, unknown_anomaly_classes)

    normal_train, normal_test = random_split(normal_idx, train_ratio)
    known_anomaly_train, known_anomaly_test = random_split(known_anomaly_idx, train_ratio)
    unknown_anomaly_test = unknown_anomaly_idx

    train_idx = np.hstack((normal_train, known_anomaly_train))

    train_idx = torch.LongTensor(train_idx)
    val_idx = train_idx

    test_idx = {
        'all': np.hstack((normal_test, known_anomaly_test, unknown_anomaly_test)),
        'known': np.hstack((normal_test, known_anomaly_test)),
        'unknown': np.hstack((normal_test, unknown_anomaly_test)),
        'normal': normal_test,
        'known_only': known_anomaly_test,
        'unknown_only': unknown_anomaly_test,
    }

    split_info = {
        'idx_train': train_idx,
        'idx_normal_train': normal_train,
        'idx_anomaly_train': known_anomaly_train,
        'idx_val': val_idx,
        'idx_test': test_idx
    }

    return split_info


def ad_split_num(labels, train_ratio, num_anomaly, class_info):
    known_anomaly = class_info['known_anomaly']
    unknown_anomaly_classes = class_info['unknown_anomaly']
    normal_classes = class_info['normal']
    print(normal_classes)

    known_anomaly_idx = np.where(labels==known_anomaly)[0].flatten()
    normal_idx = select_class_idx_in_list(labels, normal_classes)
    unknown_anomaly_idx = select_class_idx_in_list(labels, unknown_anomaly_classes)

    normal_train, normal_test = random_split(normal_idx, train_ratio)
    known_anomaly_train, known_anomaly_test = num_split(known_anomaly_idx, num_anomaly)
    unknown_anomaly_test = unknown_anomaly_idx

    train_idx = np.hstack((normal_train, known_anomaly_train))

    train_idx = torch.LongTensor(train_idx)
    val_idx = train_idx

    test_idx = {
        'all': np.hstack((normal_test, known_anomaly_test, unknown_anomaly_test)),
        'known': np.hstack((normal_test, known_anomaly_test)),
        'unknown': np.hstack((normal_test, unknown_anomaly_test)),
        'normal': normal_test,
        'known_only': known_anomaly_test,
        'unknown_only': unknown_anomaly_test,
    }

    split_info = {
        'idx_train': train_idx,
        'idx_normal_train': normal_train,
        'idx_anomaly_train': known_anomaly_train,
        'idx_val': val_idx,
        'idx_test': test_idx
    }

    return split_info


class DglDataset:
    def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None, view=None):
        dataset = FraudYelpDataset()
        graph = dataset[0]
        graph = dgl.edge_type_subgraph(graph, [view])
        graph = dgl.add_self_loop(graph)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph