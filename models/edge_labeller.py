import torch
import torch.nn as nn

class EdgeLabellerFusedOri(nn.Module):
    def __init__(self, ebd_dim, out_dim):
        super(EdgeLabellerFusedOri, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(ebd_dim, ebd_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight)
        self.linear = nn.Linear(ebd_dim, out_dim)

    def forward(self, x, y):
        ebd_fused = torch.matmul(torch.sigmoid(x), self.weight) * torch.sigmoid(y)
        # ebd_fused = torch.matmul(x, self.weight) * y
        return self.linear(ebd_fused)
