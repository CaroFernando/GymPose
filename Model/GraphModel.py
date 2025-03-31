from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling

from torch import nn
from torch.nn import functional as F

from dataclasses import dataclass
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from sklearn.metrics import f1_score


from typing import Optional

class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )
    
class MultiFeatureRBFExpansion(nn.Module):
    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        super().__init__()
        self.rbfe = RBFExpansion(vmin, vmax, bins, lengthscale)
    
    def forward(self, x):
        n_feature = x.shape[1]
        x = torch.cat([self.rbfe(x[:, i]).unsqueeze(1) for i in range(n_feature)], dim=2).squeeze(1)
        return x

class MLPExpantion(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)

class EdgeGatedGraphConv(nn.Module):
    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)
        self.dp_edges = nn.Dropout(0.2)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)
        self.dp_nodes = nn.Dropout(0.2)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()
        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))
        x = self.dp_nodes(x)
        y = self.dp_edges(y)

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)
        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)

class ALIGNN(pl.LightningModule):
    def __init__(self, config, weights):
        super().__init__()
        # print(config)
        self.config = config
        self.weights = weights
        self.pos_weight = torch.tensor([self.weights[1]], dtype=torch.float32).to('cuda')
        # Node encoder process

        self.node_embeddings = MLPLayer(
            config.node_input_features, config.hidden_features
        )

        # Edge encoder process
        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_angle_input_features,
            ),
            MLPLayer(config.edge_angle_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.edge_class_embedding = nn.Sequential(
            MLPLayer(config.edge_input_features, config.hidden_features)
        )

        self.edge_joined_embedding = nn.Sequential(
            MLPLayer(
                2 * config.hidden_features,
                config.hidden_features,
            )
        )

        self.angle_embedding = nn.Sequential(
            MultiFeatureRBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ) if config.expantion_type == 'rbfe' else MLPExpantion(3, config.triplet_input_features),
            MLPLayer(
                config.triplet_input_features * (3 if config.expantion_type == 'rbfe' else 1), 
                config.embedding_features
            ),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        # Model Layers
        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()
        self.fc = nn.Linear(config.hidden_features, 1)
        self.outact = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, num_classes=1, task="binary", average="binary")
        self.f1 = F1Score(threshold=0.5, num_classes=1, task="binary", average="binary")

    def forward(
        self, g: Tuple[dgl.DGLGraph, dgl.DGLGraph]
    ):
        g, lg = g
        lg = lg.local_var()
        g = g.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("node_features")
        x = self.node_embeddings(x)
        # initial bond features

        edge_features = g.edata.pop("edge_features")
        edge_features = self.edge_class_embedding(edge_features)

        edge_length = torch.norm(g.edata.pop("r"), dim=1)
        edge_length = self.edge_embedding(edge_length)

        y = torch.cat([edge_features, edge_length], dim=1)
        y = self.edge_joined_embedding(y)
        
        h = lg.edata.pop("h")
        z = self.angle_embedding(h)
        
        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        # print('features',features.shape)
        out = self.fc(h)
        #out = self.outact(out)
        return out
    
    def training_step(self, batch, batch_idx):
        x, lx, y = batch
        y_hat = self((x, lx))

        loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight = self.pos_weight)
        y_hat = self.outact(y_hat)
        acc = self.accuracy(y_hat, y)
        f1_s = self.f1(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1_s, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, lx, y = batch
        y_hat = self((x, lx))
        # bce loss
        loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight = self.pos_weight)
        y_hat = self.outact(y_hat)
        acc = self.accuracy(y_hat, y)
        f1_s = self.f1(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1_s, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, lx, y = batch
        y_hat = self((x, lx))
        # bce loss
        loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight = self.pos_weight)
        y_hat = self.outact(y_hat)
        acc = self.accuracy(y_hat, y)

        f1_s = self.f1(y_hat, y)
        y_hat = (y_hat >= 0.5).float()
        y_hat = y_hat.cpu().numpy()
        y = y.cpu().numpy()
        f1_score_normal = f1_score(y,y_hat)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1_s, prog_bar=True)
        self.log("test_f1_normal", f1_score_normal,  prog_bar=True)
        return loss

    def calc_lr(self, step, dim_embed, warmup_steps):
        return dim_embed**(-0.5) * min((step+1)**(-0.5), (step+1) * warmup_steps**(-1.5))

    def configure_optimizers(self):
        op = torch.optim.Adam(self.parameters(), lr = 0.1, betas=(0.9, 0.98))
        warmup_steps = 300
        sch = torch.optim.lr_scheduler.LambdaLR(op, lambda step: self.calc_lr(step, self.config.hidden_features, warmup_steps))

        return {
            "optimizer": op, 
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "step",
                "frequency": 1
            }         
        }