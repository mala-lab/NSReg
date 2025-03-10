from models.base_gnns import GraphSAGE, GAT
from models.base_nns import MLP
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        if args.encoder_type.lower() == 'gat':
            self.encoder = GAT(args.input_dim, args.hidden_dim, args.hidden_dim, args.n_layers, args.n_heads)
            print("GAT used as the encoder")
        elif args.encoder_type.lower() == 'sage':
            self.encoder = GraphSAGE(args.input_dim, args.hidden_dim, args.hidden_dim, args.n_layers, args.drop_out,
                                     output_type="ebds", adj_dropout=0.0)
            print("GraphSAGE used as the encoder")
        else:
            raise NotImplementedError("Encoder type not supported")
        self.proj = MLP(args.ebd_dim, 64, 64)

    def forward(self, x, adjs):
        ebd = self.encoder(x, adjs)
        ebd_proj = self.proj(ebd)
        return ebd_proj