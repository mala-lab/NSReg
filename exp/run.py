import argparse
from utils.io import load_npz, load_yaml, npz_data_to_pyg_graph, load_dgl_graph, dgl_data_to_pyg_graph
from os.path import join
import torch
import numpy as np
from runners.train_runner import TrainRunner
from datetime import datetime
import time
import torch_geometric.transforms as T
from dgl.data.utils import load_graphs
from utils.io import make_pyg_graph_dgl

# Different experiment mode
def do_exp(graph, labels, dset_info, mode="single", tau_lower=0.00, tau_upper=0.05, args=None):
    print("Treat any class whose proportion %f <= %f as anomaly classes" % (tau_lower, tau_upper))
    anomaly_class_idx = np.where((dset_info['class_per'] <= tau_upper) & (dset_info['class_per'] >= tau_lower))[0]
    print("Total anomaly percentage %.2f%%" % (dset_info['class_per'][anomaly_class_idx].sum() * 100))
    if args.dset_name in ["ogbn-proteins"]:
        anomaly_class_idx = dset_info['class_idx'][anomaly_class_idx]
    print("anomaly classes: %s" % str(anomaly_class_idx.tolist()))
    assert mode in ["run", "sup", "visualise"]
    if mode == "run" or mode == "visualise":
        print("Training with one fully labelled class...")
        for idx in anomaly_class_idx:
            # if idx in [0, 1, 3, 8, 9, 12, 14]:
            #     continue
            class_name = dset_info['class_names'][idx] if dset_info['class_names'] is not None else str(idx)
            print("using class %d: %s as known anomalies" % (idx, class_name))
            # print("Known anomalies: %.2f%%" % (dset_info['class_per'][idx]*100))
            anomaly_info = {'known_anomaly': idx,
                            'unknown_anomaly': [i for i in anomaly_class_idx if i != idx],
                            'normal': [i for i in dset_info['class_idx'] if i not in anomaly_class_idx],
                            'all_anomaly': anomaly_class_idx}
            runner = TrainRunner(graph, labels, dset_info, anomaly_info, args)
            runner.train()
    else:
        raise NotImplementedError("Invalid node!")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_dir", type=str, default=<set to project directory>)
    parser.add_argument("--meta_config_fn", type=str, default="mag_cs") # set to data names 
    parser.add_argument("--config_dir", type=str, default="mag_cs") # set to data names 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_id", type=str, default=None)
    args = parser.parse_args()

    print("Using train.py")

    if args.meta_config_fn is not None:
        config_fns = load_yaml(join(args.proj_dir, "exp", "config", args.config_dir, args.meta_config_fn + ".yaml"))
        for key in config_fns.keys():
            config = load_yaml(config_fns[key])
            args = argparse.Namespace(**{**vars(args), **config})
    else:
        raise RuntimeError("Meta file not found!")

    print(args)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    print("timestamp: %s" % timestamp)
    args.ts = timestamp

    print("using device %s" % args.device)
    args.device = torch.device('cuda:' + str(args.device))

    if args.dset_name in ["amz_computers", "amz_photo", "mag_cs"]:
        data = load_npz(args.dset_fn)
        # graph, ad_labels, dset_info = npz_data_to_pyg_graph(data)
        graph, class_labels, dset_info = npz_data_to_pyg_graph(data)
    elif args.dset_name in ["yelp"]:
        x_all, adj = load_dgl_graph(args.dset_name, homo=1, view=args.view)
        class_labels = torch.load(join(args.saved_idx_dir, args.dummy_class_fn))
        graph, dset_info = dgl_data_to_pyg_graph(x_all, adj, class_labels)
    
    t_start = time.time()
    print("time start:", t_start)
    do_exp(graph, class_labels, dset_info, mode=args.mode, args=args, tau_lower=args.tau_lower, tau_upper=args.tau_upper)
    t_end = time.time()
    print("time end:", t_end)
    print("Total time: %.2f" % (t_end - t_start))


if __name__ == "__main__":
    run()