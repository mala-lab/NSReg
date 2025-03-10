from utils.optim import get_optimiser, get_scheduler, get_lr

import torch
import pickle
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

from sklearn.manifold import TSNE

import os
from os.path import join

import warnings

from torch_geometric.loader import NeighborSampler
from models.base_nns import MLP

from models.base_gnns import GraphSAGE
# from models.loss import DeviationLoss
from models.edge_labeller import EdgeLabellerFusedOri
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.ops import get_edge_mask, subgraph_or, remove_edges, make_fully_connected

import torch_geometric.utils as pyg_utils

from torch_geometric.utils import structured_negative_sampling
from sklearn.decomposition import PCA


class TrainRunner:
    def __init__(self, graph, labels, dset_info, anomaly_info, args):
        self.x_all = graph.x.float()
        self.edge_index = graph.edge_index
        self.device = args.device
        self.labels = torch.tensor(labels.flatten()).squeeze()
        self.dset_info = dset_info
        self.anomaly_info = anomaly_info
        self.args = args

        self.create_ckpt_dir()
        # self.create_plot_dir()

        sampler_kwargs = {
            'sampling_size': self.args.sampling_sizes,
            'batch_size': self.args.batch_size,
            'fs': self.args.fs,
            'fs_ratio': self.args.fs_ratio,
        }

        if args.use_recorded_split:
            path = join(args.saved_idx_dir, args.split_fn)
            n_known = int(args.split_fn.split('_')[-1][:-4])
            print("using idx from: %s - num known anomalies %d" % (args.split_fn, n_known))
            with open(path, 'rb') as f:
                self.split_info = pickle.load(f)[anomaly_info['known_anomaly']]
            assert n_known == args.num_train_anomaly
        # else:
        #     self.split_info = ad_split_num(self.labels, args.fs_ratio, args.num_train_anomaly, anomaly_info)
        
        print("num known anomalies: %d" % self.split_info['idx_anomaly_train'].shape[0])

        # Models and losses
        if self.args.loss == 'bce':
            self.encoder = GraphSAGE(args.input_dim, args.hidden_dim, args.hidden_dim, args.n_layers, args.drop_out,
                                         output_type="ebds", adj_dropout=0.0)
            print("GRAPHSAGE used as the encoder")
            self.clf_criterion = torch.nn.BCEWithLogitsLoss()
            self.clf = MLP(args.ebd_dim, 32, 1)
        elif self.args.loss == 'dev':
            print("using deviation loss...")
            self.encoder = GraphSAGE(args.input_dim, args.hidden_dim, args.hidden_dim, args.n_layers, args.drop_out,
                                     output_type="ebds", adj_dropout=0.0)
            self.clf_criterion = DeviationLoss()
            self.clf = torch.nn.Linear(args.hidden_dim, 1)
        else:
            raise NotImplementedError("Loss is not supported!")

        self.edge_labeller = EdgeLabellerFusedOri(self.args.ebd_dim, 1)

        self.proj = MLP(args.ebd_dim, args.ebd_dim, args.ebd_dim)
        self.edge_index.to(self.device)

        # Dataloaders
        print("building the dataloader ...")
        self.dataloader = NeighborSampler(self.edge_index,
                                           node_idx=None,
                                           sizes=self.args.sampling_sizes,
                                           batch_size=50000,
                                           shuffle=False,
                                           drop_last=False,
                                           pin_memory=True)

        self.en_optimiser = get_optimiser(self.args.optimiser, self.encoder.parameters(), lr=self.args.lbl_lr,
                                          weight_decay=self.args.weight_decay)
        self.lbl_optimiser = get_optimiser(self.args.optimiser, self.edge_labeller.parameters(), lr=self.args.lbl_lr,
                                           weight_decay=self.args.weight_decay)
        self.mlp_optimiser = get_optimiser(self.args.optimiser, self.clf.parameters(), lr=self.args.lr,
                                           weight_decay=self.args.weight_decay)
        self.proj_optimiser = get_optimiser(self.args.optimiser, self.proj.parameters(), lr=self.args.lbl_lr,
                                            weight_decay=self.args.weight_decay)

        self.train_binary_ad_labels, self.eval_binary_ad_labels = np.zeros(shape=self.labels.shape[0]), \
            np.zeros(shape=self.labels.shape[0])

        for i in self.anomaly_info['all_anomaly']:
            self.eval_binary_ad_labels[np.where(self.labels == i)[0]] = 1
        self.eval_binary_ad_labels = torch.from_numpy(self.eval_binary_ad_labels)

        if not self.args.contaminated:
            self.train_binary_ad_labels = self.eval_binary_ad_labels
        else:
            for i in [self.anomaly_info['known_anomaly']]:
                self.train_binary_ad_labels[np.where(self.labels == i)[0]] = 1
            self.train_binary_ad_labels = torch.from_numpy(self.train_binary_ad_labels)
            print("Number of known anomalies: %d" % self.train_binary_ad_labels.sum())

        # Label generation checked.

        if self.args.dset_name == 'yelp':
            assert self.eval_binary_ad_labels.sum() == (self.labels!=0).sum()
            print("assertion passed...")

        self.pos_edge_mask, self.neg_edge_mask = get_edge_mask(self.edge_index, self.eval_binary_ad_labels)
        self.n_nodes = self.x_all.size(0)

        self.lbl_criterion = torch.nn.BCEWithLogitsLoss()
        self.single_layer = self.args.n_layers == 1

        if not self.args.train_labeller:
            return
        self.pos_edges_train = \
        pyg_utils.subgraph(self.split_info['idx_normal_train'], self.edge_index, num_nodes=self.n_nodes)[0]

        if self.args.train_labeller:
            self.normal_train_unconnected = make_fully_connected(torch.tensor(self.split_info['idx_normal_train']))
            self.normal_train_unconnected = remove_edges(self.normal_train_unconnected , self.pos_edges_train)
            self.normal_train_unconnected = remove_edges(self.normal_train_unconnected, self.pos_edges_train[[1, 0]])

        self.pos_edges_train_sampling = torch.tensor(
            [[i, j] for i in self.split_info['idx_normal_train'] for j in self.split_info['idx_normal_train'] if
             i != j], dtype=torch.long).t().contiguous()

        # Any edges attached to more than one known anomaly node is labelled as negative
        self.neg_edges_train = subgraph_or(self.split_info['idx_anomaly_train'], self.edge_index,
                                           num_nodes=self.n_nodes)
        print("num train pos edges: %d  num train neg edges: %d" % (
        self.pos_edges_train.size(1), self.neg_edges_train.size(1)))
        # self.lbl_criterion = torch.nn.L1Loss(reduction='mean')

        self.pos_edges_test_all = \
        pyg_utils.subgraph(self.split_info['idx_test']['unknown'], self.edge_index[:, self.pos_edge_mask],
                           num_nodes=self.n_nodes)[0]
        self.neg_edges_test_all = \
        pyg_utils.subgraph(self.split_info['idx_test']['unknown'], self.edge_index[:, self.neg_edge_mask],
                           num_nodes=self.n_nodes)[0]

        self.edge_labels = torch.zeros(self.edge_index.shape[1])
        self.edge_labels[self.neg_edge_mask] = 1

        self.src_test = torch.cat((self.pos_edges_test_all[0], self.neg_edges_test_all[0]))
        self.tar_test = torch.cat((self.pos_edges_test_all[1], self.neg_edges_test_all[1]))
        self.test_edge_labels = torch.cat(
            (torch.zeros(self.pos_edges_test_all.size(1)), torch.ones(self.neg_edges_test_all.size(1))))

    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        self.clf.to(self.device), self.encoder.to(self.device), self.edge_labeller.to(self.device), self.proj.to(
            self.device)
        self.labels.to(self.device)
        self.clf_criterion.to(self.device)
        self.lbl_criterion.to(self.device)
        self.labels = self.labels.long()

        # n_batches = len(self.train_loader)
        print("NSReg Training ...")

        # print("%3d batches per epoch" % n_batches)
        if self.args.use_batch:
            # batch_num_normal = self.split_info['idx_normal_train'].shape[0] if self.split_info['idx_normal_train'].shape[0] < self.args.batch_size else self.args.batch_size
            # clf_weight = (batch_num_normal/ self.args.num_train_anomaly) if self.args.oversample else 1.0
            # print("Batch mode: enabled  batch_size=%d  weight_factor=%.4f" % (batch_num_normal, clf_weight))
            print("Batch mode: enabled", end=" ")
            if self.args.oversample:
                print("oversample")
            else:
                print("original propertion")
        else:
            print("Batch mode: disabled")

        for epoch in range(self.args.n_epochs):
            self.clf.train(), self.encoder.train(), self.edge_labeller.train(), self.proj.train()
            if epoch % self.args.eval_every == 0:
                epoch_str = "Epoch %2d" % epoch
                print(epoch_str, end=" ")

            # if epoch % self.args.plot_every == 0 and epoch != 0:
            #     batch_str = "bj_"+self.args.batch_id if self.args.batch_id is not None else ""
            #     fn = join(self.args.plot_dir, batch_str, self.args.ts, "pre_ano%d_epo_%d_v1.png" % (self.anomaly_info['known_anomaly'], epoch))
            #     self.visualise(self.encoder, self.proj, None, fn)

            self.mlp_optimiser.zero_grad(), self.en_optimiser.zero_grad(), self.lbl_optimiser.zero_grad(), self.proj.zero_grad()

            all_ebds_enc = self.get_all_embedding(self.encoder, self.dataloader, self.x_all, len(self.dataloader))
            all_ebds = self.proj(all_ebds_enc)
            # pred = self.clf(self.proj(all_ebds))
            pred = self.clf(all_ebds)

            if not self.args.use_batch:
                clf_loss = self.clf_criterion(pred[self.split_info['idx_train']],
                                              self.train_binary_ad_labels[self.split_info['idx_train']].unsqueeze(
                                                  1).float().to(self.device))
            else:
                idx_anom_train = self.split_info['idx_anomaly_train']
                idx_normal_train = self.split_info['idx_normal_train']
                if idx_normal_train.shape[0] > self.args.batch_size:
                    batch_idx_train = idx_normal_train[torch.randperm(idx_normal_train.shape[0])[:self.args.batch_size]]
                else:
                    batch_idx_train = idx_normal_train

                if self.args.oversample:
                    clf_loss_anom = self.clf_criterion(pred[idx_anom_train],
                                                       self.train_binary_ad_labels[idx_anom_train].unsqueeze(
                                                           1).float().to(self.device))
                    clf_loss_normal = self.clf_criterion(pred[batch_idx_train],
                                                         self.train_binary_ad_labels[batch_idx_train].unsqueeze(
                                                             1).float().to(self.device))
                    clf_loss = clf_loss_anom + clf_loss_normal
                else:
                    if self.args.augment is None:
                        batch_idx = torch.cat((torch.from_numpy(idx_anom_train), torch.from_numpy(batch_idx_train)))
                        clf_loss = self.clf_criterion(pred[batch_idx],
                                                    self.train_binary_ad_labels[batch_idx].unsqueeze(1).float().to(
                                                        self.device))
                    elif self.args.augment == "mixup":
                        anom_ebds, norm_ebds = all_ebds_enc[idx_anom_train], all_ebds_enc[batch_idx_train]
                        rand_norm_idx = torch.randperm(norm_ebds.shape[0])
                        rand_anom_idx = torch.randperm(anom_ebds.shape[0])
                        n_mixup = anom_ebds.shape[0]
                        rand_anom_idx = rand_anom_idx.repeat((n_mixup // anom_ebds.shape[0] + 1))
                        alpha = 0.1
                        # mixed_up_ebds = alpha * anom_ebds[rand_anom_idx[:n_mixup]] + (1 - alpha) * norm_ebds[rand_norm_idx[:n_mixup]]
                        mixed_up_ebds = alpha * anom_ebds[rand_anom_idx[:n_mixup]] + (1 - alpha) * all_ebds_enc[torch.randperm(all_ebds_enc.size(0))[:n_mixup]]
                        mixed_up_pred = self.clf(self.proj(mixed_up_ebds))


                        #mix up anomalies
                        rand_anom_idx = torch.randperm(anom_ebds.shape[0])
                        mix_up_anom_ebds = (anom_ebds + anom_ebds[rand_anom_idx]) / 2
                        mix_up_anom_pred = self.clf(self.proj(mix_up_anom_ebds))

                        # anom_ebds, norm_ebds = all_ebds[idx_anom_train], all_ebds[batch_idx_train]
                        # n_mixup = norm_ebds.shape[0]
                        # rand_anom_idx = torch.randperm(anom_ebds.shape[0])
                        # rand_anom_idx = rand_anom_idx.repeat((norm_ebds.shape[0] // anom_ebds.shape[0] + 1))
                        # n_mixup = norm_ebds.shape[0]
                        # alpha = 0.5
                        # # mixed_up_ebds = alpha * anom_ebds[rand_anom_idx[:n_mixup]] + (1 - alpha) * all_ebds[torch.randperm(all_ebds.shape[0])[:n_mixup]]
                        # mixed_up_ebds = alpha * all_ebds[torch.randperm(all_ebds.shape[0])[n_mixup:2 * n_mixup]]+ (1 - alpha) * all_ebds[torch.randperm(all_ebds.shape[0])[:n_mixup]]
                        # mixed_up_pred = self.clf(mixed_up_ebds)
                        batch_idx = torch.cat((torch.from_numpy(idx_anom_train), torch.from_numpy(batch_idx_train)))
                        # clf_loss = self.clf_criterion(torch.cat((pred[batch_idx], mixed_up_pred, mix_up_anom_pred)),
                        #                               torch.cat((self.train_binary_ad_labels[batch_idx].unsqueeze(1).float().to(self.device),
                        #                                          1 * torch.ones(n_mixup).unsqueeze(1).float().to(self.device),
                        #                                          torch.ones(mix_up_anom_ebds.size(0)).unsqueeze(1).float().to(self.device))))
                        # clf_loss = self.clf_criterion(torch.cat((pred[batch_idx], mixed_up_pred)),
                        #                               torch.cat((self.train_binary_ad_labels[batch_idx].unsqueeze(1).float().to(self.device),
                        #                               1 * torch.ones(n_mixup).unsqueeze(1).float().to(self.device))))


                        all_pred = torch.cat((pred[batch_idx], mixed_up_pred, mix_up_anom_pred))
                        all_labels = torch.cat((self.train_binary_ad_labels[batch_idx].unsqueeze(1).float().to(self.device), \
                                                1.0 * torch.ones(n_mixup).unsqueeze(1).float().to(self.device), \
                                                torch.ones(mix_up_anom_ebds.size(0)).unsqueeze(1).float().to(self.device)))
                        clf_loss = self.clf_criterion(all_pred, all_labels)
                    elif self.args.augment == "x_mixup":
                        # swap anomalies with random normal samples
                        self.x_all_copy = self.x_all.clone()
                        n_nodes = self.x_all.shape[0]
                        n_shuffle = int(0.7 * n_nodes) 
                        shuffle_idx = torch.randperm(n_nodes)[:n_shuffle]
                        shuffle_idx_rnd = shuffle_idx[torch.randperm(shuffle_idx.shape[0])]
                        self.x_all_copy[shuffle_idx] = self.x_all_copy[shuffle_idx_rnd]
                        batch_idx = torch.cat((torch.from_numpy(idx_anom_train), torch.from_numpy(batch_idx_train)))
                        all_ebds_enc = self.get_all_embedding(self.encoder, self.dataloader, self.x_all_copy, len(self.dataloader))
                        aug_ebds  = all_ebds_enc[shuffle_idx]
                        aug_pred = self.clf(self.proj(aug_ebds))[torch.randperm(aug_ebds.size(0))[:50]]

                        all_pred = torch.cat((pred[batch_idx], aug_pred))
                        all_labels = torch.cat((self.train_binary_ad_labels[batch_idx].unsqueeze(1).float().to(self.device), \
                                               0.9 * torch.ones(50).unsqueeze(1).float().to(self.device)))
                        clf_loss = self.clf_criterion(all_pred, all_labels)
            
            if not self.args.train_labeller:
                clf_loss.backward()
                self.mlp_optimiser.step()
                self.proj_optimiser.step()
                self.en_optimiser.step()
                if epoch % self.args.eval_every == 0:
                    print("loss=%.4f" % clf_loss)
                if epoch % self.args.eval_every == 0:
                    self.val()

                if epoch % self.args.save_every == 0:
                    self.save_ckpt(epoch)
                continue

            if self.args.edge_sample_mode == "original":
                pos_neg_samples = structured_negative_sampling(self.pos_edges_train, num_nodes=self.x_all.size(0),
                                                               contains_neg_self_loops=False)
                n_pos_edge = self.pos_edges_train.size(1)
                
                n_normal_unconnected = self.normal_train_unconnected.size(1)
                normal_unconnected_rand = self.normal_train_unconnected[:,torch.randperm(n_normal_unconnected)[:n_pos_edge]]

                all_src_idx = torch.cat((self.pos_edges_train[0], self.pos_edges_train[0], normal_unconnected_rand[0]))       
                all_tar_idx = torch.cat((pos_neg_samples[2], self.pos_edges_train[1], normal_unconnected_rand[1]))
                edge_labels = torch.cat((0.0 * torch.ones(pos_neg_samples[2].size(0), dtype=torch.long),
                                         1 * torch.ones(self.pos_edges_train.size(1), dtype=torch.long),
                                         self.args.nnu_alpha * torch.ones(n_pos_edge, dtype=torch.long)))
            elif self.args.edge_sample_mode == "single_label":
                pos_neg_samples = structured_negative_sampling(self.pos_edges_train, num_nodes=self.x_all.size(0),
                                                               contains_neg_self_loops=False)
                n_pos_edge = self.pos_edges_train.size(1)
                
                n_normal_unconnected = self.normal_train_unconnected.size(1)
                normal_unconnected_rand = self.normal_train_unconnected[:,torch.randperm(n_normal_unconnected)[:n_pos_edge]]

                all_src_idx = torch.cat((self.pos_edges_train[0], self.pos_edges_train[0]))       
                all_tar_idx = torch.cat((pos_neg_samples[2], self.pos_edges_train[1]))
                edge_labels = torch.cat((0.0 * torch.ones(pos_neg_samples[2].size(0), dtype=torch.long),
                                         1 * torch.ones(self.pos_edges_train.size(1), dtype=torch.long)))
            else:
                raise NotImplementedError("neg sampling mode is not supported!")
            assert all_src_idx.size() == all_tar_idx.size()
            rand_edge_idx = torch.arange(all_src_idx.size(0))[torch.randperm(all_src_idx.size(0))][:512]
            edge_pred = self.edge_labeller(all_ebds[all_src_idx[rand_edge_idx]], all_ebds[all_tar_idx[rand_edge_idx]])

            if self.args.edge_aug:
                self.x_all_copy = self.x_all.clone()
                n_nodes = self.x_all.shape[0]
                n_shuffle = int(0.7 * n_nodes) 
                shuffle_idx = torch.randperm(n_nodes)[:n_shuffle]
                shuffle_idx_rnd = shuffle_idx[torch.randperm(shuffle_idx.shape[0])]
                self.x_all_copy[shuffle_idx] = self.x_all_copy[shuffle_idx_rnd]
                all_ebds_enc = self.proj(self.get_all_embedding(self.encoder, self.dataloader, self.x_all_copy, len(self.dataloader)))
                aug_ebds  = all_ebds_enc[shuffle_idx]
                aug_ebds_rnd = aug_ebds[torch.randperm(aug_ebds.shape[0])][: 1 * self.pos_edges_train[0].size(0)]
                edge_pred_aug = self.edge_labeller(all_ebds[self.pos_edges_train[0]], aug_ebds_rnd)
                if edge_pred_aug.size(0) > 128:
                    edge_pred_aug = edge_pred_aug[torch.randperm(edge_pred_aug.shape[0])][:128]
                lbl_loss = self.lbl_criterion(torch.cat((edge_pred, edge_pred_aug)),\
                                              torch.cat((edge_labels[rand_edge_idx].unsqueeze(1).float().to(self.device), 
                                                         0.0 * torch.ones(edge_pred_aug.size(0), 1).to(self.device)
                                                         )))
            else:
                lbl_loss = self.lbl_criterion(edge_pred, edge_labels[rand_edge_idx].unsqueeze(1).float().to(self.device))

            loss = clf_loss + 1.0 * lbl_loss
            loss.backward()
            if epoch % self.args.eval_every == 0:
                print("loss=%.4f, clf_loss=%.4f, lbl_loss=%.4f" % (loss, clf_loss, lbl_loss))

            self.lbl_optimiser.step()
            self.mlp_optimiser.step()
            self.proj_optimiser.step()
            self.en_optimiser.step()

            if epoch % self.args.eval_every == 0:
                self.val()

    def val(self):
        self.encoder.eval(), self.clf.eval(), self.proj.eval()
        xs = []

        with torch.no_grad():
            if self.args.encoder_type.lower() in ["sage"]:
                for batch_size, n_id, adjs in self.dataloader:
                    if self.args.n_layers == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    out = self.encoder(self.x_all[n_id].to(self.device), adjs)
                    # x  = F.relu(out)
                    out = self.clf(self.proj(out))
                    xs.append(out.cpu())

                x_all = torch.cat(xs, dim=0)
            else:
                x_all = self.model(self.x_all.to(self.device), self.edge_index.to(self.device))
            # y_pred = torch.exp(x_all).permute(1, 0)[1]

            y_pred = torch.sigmoid(x_all)

        pred = y_pred.cpu()
        split_info = self.split_info
        labels = self.eval_binary_ad_labels.cpu()
        test_dict = self.split_info['idx_test']
        test_all, test_known, test_un = test_dict['all'], test_dict['known'], test_dict['unknown']
        print("test_all: ", test_all.shape[0])
        auroc_all, aupr_all = roc_auc_score(labels, pred), average_precision_score(labels, pred)
        auroc_train, aupr_train = roc_auc_score(labels[split_info['idx_train']],
                                                pred[split_info['idx_train']]), average_precision_score(
            labels[split_info['idx_train']], pred[split_info['idx_train']])
        auroc_test_all, aupr_test_all = roc_auc_score(labels[test_all], pred[test_all]), average_precision_score(
            labels[test_all], pred[test_all])
        auroc_test_known, aupr_test_known = roc_auc_score(labels[test_known],
                                                          pred[test_known]), average_precision_score(labels[test_known],
                                                                                                     pred[test_known])
        auroc_test_un, aupr_test_un = roc_auc_score(labels[test_un], pred[test_un]), average_precision_score(
            labels[test_un], pred[test_un])
        
        print("auroc | all=%.4f  train=%.4f  test_all=%.4f  test_known=%.4f  test_unknown=%.4f" %
              (auroc_all, auroc_train, auroc_test_all, auroc_test_known, auroc_test_un))
        print("aupr | all=%.4f  train=%.4f  test_all=%.4f  test_known=%.4f  test_unknown=%.4f" %
              (aupr_all, aupr_train, aupr_test_all, aupr_test_known, aupr_test_un))

        return None
    
    def plot_decision_boundary_tnse(self, x, n_samples=500):
        idx_known = torch.from_numpy(self.split_info['idx_test']['known'])
        idx_unknown = torch.from_numpy(self.split_info['idx_test']['unknown'])
        idx_normal_test = torch.from_numpy(self.split_info['idx_test']['normal'])

        idx_known_sample = idx_known[torch.randperm(idx_known.size(0))[:n_samples]]
        idx_unknown_sample = idx_unknown[torch.randperm(idx_unknown.size(0))[:n_samples]]
        idx_normal_test_sample = idx_normal_test[torch.randperm(idx_normal_test.size(0))[:n_samples]]

        idx_all_sample = torch.cat([idx_known_sample, idx_unknown_sample, idx_normal_test_sample])
        plot_labels = torch.cat([torch.ones(n_samples), 2 * torch.zeros(n_samples), torch.zeros(n_samples)])

        x_samples = x[idx_all_sample].cpu().numpy()
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(x_samples)

        dim = 3
        # construct a mesh grid
        grid_shape = [10] * dim
        print(grid_shape)
        grids = []

        for i in range(dim):
            axis_min = x[:, i].min() - 1
            axis_max = x[:, i].max() + 1
            grid_i = np.linspace(axis_min, axis_max, grid_shape[i])
            grids.append(grid_i)
        
        grid = np.meshgrid(*grids)
        grid = np.stack(grid, axis=-1).reshape(-1, 10)
        grid = torch.tensor(grid, dtype=torch.float32)

        with torch.no_grad():
            Z = self.linear_predict(self.clf.layers[2], grid)
            Z = Z.reshape(grid_shape)
        
        fig, ax = plt.subplots()
        ax.contourf(X_tsne[:, 0], X_tsne[:, 1], Z, alpha=0.4)
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=plot_labels.numpy(), alpha=0.8)
        ax.set_xlabel('t-SNE component 1')
        ax.set_ylabel('t-SNE component 2')
        ax.set_title('Decision boundary of a PyTorch classifier using t-SNE')
        plt.show()


    def plot_decision_boundry(self, x):
        idx_anom = torch.arange(x.size(0))[self.eval_binary_ad_labels == 1]
        idx_normal =  torch.arange(x.size(0))[self.eval_binary_ad_labels == 0]
        rand_idx = torch.randperm(idx_normal.size(0))[:idx_anom.size(0)]
        idx = torch.cat([idx_anom, idx_normal[rand_idx]])
        x = x.cpu().numpy()[idx]
        pca = PCA(n_components=2)
        x_2d = pca.fit_transform(x)

        x_min, x_max = x_2d[:, 0].min() - 1, x_2d[:, 0].max() + 1
        y_min, y_max = x_2d[:, 1].min() - 1, x_2d[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
        
        grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
        grid_points_original_space = pca.inverse_transform(grid_points_2d)
        
        Z = self.linear_predict(self.clf.layers[2], grid_points_original_space)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(x_2d[:, 0], x_2d[:, 1], c=self.eval_binary_ad_labels[idx], edgecolors='k', marker='o', s=50)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Decision Boundary of Neural Network Classifier (PCA)')
        plt.show()
        print(os.getcwd())
        # plt.savefig('plot/boundary/%s/%s_anom_%d.png' % (self.args.dset_name, self.args.ts, self.anomaly_info['known_anomaly']))
        plt.savefig(join(self.args.proj_dir, 'plot', 'boundary', self.args.dset_name, '%s_anom_%d.png' % (self.args.ts, self.anomaly_info['known_anomaly'])))
        plt.close()

    def plot_boundary(self):
        self.load_ckpt()
        self.encoder.to(self.device), self.clf.to(self.device), self.proj.to(self.device)
        self.encoder.eval(), self.clf.eval(), self.proj.eval()
        xs = []

        with torch.no_grad():
            if self.args.encoder_type.lower() in ["gat", "sage"]:
                for batch_size, n_id, adjs in self.dataloader:
                    if self.args.n_layers == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    out = self.encoder(self.x_all[n_id].to(self.device), adjs)
                    # x  = F.relu(out)
                    out = self.clf.get_last_layer(self.proj(out))
                    xs.append(out.cpu())

                ebds = torch.cat(xs, dim=0).cpu()
            else:
                x_all = self.model(self.x_all.to(self.device), self.edge_index.to(self.device))
            # y_pred = torch.exp(x_all).permute(1, 0)[1]
        
        print(ebds.shape)
        # self.plot_decision_boundry(ebds)
        self.plot_decision_boundary_tnse(ebds)


    def get_all_embedding(self, model, loader, x_all, n_batches):
        model.to(self.device)

        loader_iter = iter(loader)
        for bn in range(len(loader)):
            if bn == n_batches:
                break

            ebds = self.single_pass(model, x_all, loader_iter, self.single_layer)

            if bn == 0:
                all_ebds = ebds
            else:
                all_ebds = torch.cat((all_ebds, ebds), 0)
        return all_ebds

    def visualise(self, encoder, proj, clf, kw, save=True):
        # 'idx_train': train_idx,
        # 'idx_normal_train': normal_train,
        # 'idx_anomaly_train': known_anomaly_train,
        # 'idx_val': val_idx,
        # 'idx_test': test_idx

        print("Visualising the latent space...")
        encoder.eval(), proj.eval()

        with torch.no_grad():
            all_ebds = proj(self.get_all_embedding(encoder, self.dataloader, self.x_all, len(self.dataloader)))

        print("Applying TSNE...")
        all_ebds = all_ebds.cpu().numpy()
        ebd_transformed = TSNE(n_components=2, init='random', n_jobs=16, random_state=0).fit_transform(all_ebds)

        ebd_transformed_t = np.transpose(ebd_transformed)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

        all_labels = self.eval_binary_ad_labels
        colors = ["green", "red", "green", "red", "blue"]
        markers = ['.', '.', '.', '^', 'x']
        names = ["norm_train", "known_train", "norm_test", "known_test", "unknown"]

        ax = ax1
        train_normal_idx = self.split_info['idx_normal_train']
        ax.scatter(ebd_transformed_t[0][train_normal_idx], ebd_transformed_t[1][train_normal_idx], color=colors[0],
                   marker=markers[0], label=names[0], s=5)

        train_anom_idx = self.split_info['idx_anomaly_train']
        ax.scatter(ebd_transformed_t[0][train_anom_idx], ebd_transformed_t[1][train_anom_idx], color=colors[1],
                   marker=markers[1], label=names[1])

        test_normal_idx = self.split_info['idx_test']['normal']
        ax.scatter(ebd_transformed_t[0][test_normal_idx], ebd_transformed_t[1][test_normal_idx], color=colors[2],
                   marker=markers[2], label=names[2], s=0)

        test_known_idx = self.split_info['idx_test']['known_only']
        ax.scatter(ebd_transformed_t[0][test_known_idx], ebd_transformed_t[1][test_known_idx], color=colors[3],
                   marker=markers[3], label=names[3], s=0)

        test_unknown_idx = self.split_info['idx_test']['unknown_only']
        ax.scatter(ebd_transformed_t[0][test_unknown_idx], ebd_transformed_t[1][test_unknown_idx], color=colors[4],
                   marker=markers[4], label=names[4], s=0)
        ax.legend(loc='center right', fancybox=True, ncol=1, fontsize='small', bbox_to_anchor=(-0.025, 0.5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax = ax2
        train_normal_idx = self.split_info['idx_normal_train']
        ax.scatter(ebd_transformed_t[0][train_normal_idx], ebd_transformed_t[1][train_normal_idx], color=colors[0],
                   marker=markers[0], label=names[0], s=0)

        train_anom_idx = self.split_info['idx_anomaly_train']
        ax.scatter(ebd_transformed_t[0][train_anom_idx], ebd_transformed_t[1][train_anom_idx], color=colors[1],
                   marker=markers[1], label=names[1], s=0)

        test_normal_idx = self.split_info['idx_test']['normal']
        ax.scatter(ebd_transformed_t[0][test_normal_idx], ebd_transformed_t[1][test_normal_idx], color=colors[2],
                   marker=markers[2], label=names[2], s=2.5)

        test_known_idx = self.split_info['idx_test']['known_only']
        ax.scatter(ebd_transformed_t[0][test_known_idx], ebd_transformed_t[1][test_known_idx], color=colors[3],
                   marker=markers[3], label=names[3], s=6)

        test_unknown_idx = self.split_info['idx_test']['unknown_only']
        ax.scatter(ebd_transformed_t[0][test_unknown_idx], ebd_transformed_t[1][test_unknown_idx], color=colors[4],
                   marker=markers[4], label=names[4], s=6)
        ax.legend(loc='center right', fancybox=True, ncol=1, fontsize='small', bbox_to_anchor=(1.3, 0.5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        fig.suptitle(kw)

        fp = kw
        # out_dir = self.out_paths['plot_dir'] + "_" + self.out_paths['timestamp']
        #
        # if not os.path.isdir(out_dir):
        #     os.makedirs(out_dir)
        # else:
        #     warnings.warn("Plot dir existed already")
        #
        plt.savefig(fp)
        plt.close()
        encoder.train(), proj.train()
        return fig

    def single_pass(self, model, x_all, loader, single_layer):
        _, n_id, adjs = next(loader)

        if single_layer:
            adjs = [adjs]

        adjs = [adj.to(self.device) for adj in adjs]

        ebds = model(x_all[n_id].to(self.device), adjs)

        return ebds
    
    def load_ckpt(self):
        ckpt_fn = "pre_ano%d_epo_%d.pt" % (self.anomaly_info['known_anomaly'], 200)
        state_dict = torch.load(join(self.args.ckpt_dir, self.args.ckpt_ts, ckpt_fn))
        self.encoder.load_state_dict(state_dict['encoder'])
        self.proj.load_state_dict(state_dict['proj'])
        self.edge_labeller.load_state_dict(state_dict['labeller'])
        self.clf.load_state_dict(state_dict['clf']) 


    def save_ckpt(self, epoch):
        state_dict = {
            'encoder': self.encoder.state_dict(),
            'proj': self.proj.state_dict(),
            'labeller': self.edge_labeller.state_dict(),
            'clf': self.clf.state_dict(),
        }

        batch_str = "bj_" + self.args.batch_id if self.args.batch_id is not None else ""

        fn = join(self.args.out_dir, batch_str, self.args.ts,
                  "pre_ano%d_epo_%d.pt" % (self.anomaly_info['known_anomaly'], epoch))
        print("saving %s" % fn)
        torch.save(state_dict, fn)

    def create_ckpt_dir(self):
        if self.args.batch_id is not None:
            ckpt_dir = join(self.args.out_dir, "bj_" + self.args.batch_id, self.args.ts)
        else:
            ckpt_dir = join(self.args.out_dir, self.args.ts)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=False)

    def create_plot_dir(self):
        if self.args.batch_id is not None:
            plot_dir = join(self.args.plot_dir, "bj_" + self.args.batch_id, self.args.ts)
        else:
            plot_dir = join(self.args.plot_dir, self.args.ts)

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=False)