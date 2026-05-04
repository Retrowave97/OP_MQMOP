import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable

from deepchem.models.losses import Loss, L2Loss
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.optimizers import Optimizer, LearningRateSchedule
from PIL import Image, ImageOps
from rdkit import Chem
from rdkit.Chem import Draw
import torchvision

from openpom.layers.pom_ffn import CustomPositionwiseFeedForward
from openpom.utils.loss import CustomMultiLabelLoss
from openpom.utils.optimizer import get_optimizer
from openpom.models.mpnn_pom import MPNNPOM

try:
    import dgl
    from dgl import DGLGraph
    from dgl.nn.pytorch import Set2Set
    # from openpom.layers.pom_mpnn_gnn import CustomMPNNGNN
    from openpom.layers.pom_mpnn_gnn_my import CustomMPNNGNN_my, CustomMPNNGNN_my_v1, CustomMPNNGNN_my_v2, CustomMPNNGNN_my_v3, CustomMPNNGNN_my_v4, CustomMPNNGNN_my_v4_C2, CustomMPNNGNN_my_v5, CustomMPNNGNN_my_v6, CustomMPNNGNN_my_v7, CustomMPNNGNN_my_v8
except (ImportError, ModuleNotFoundError):
    raise ImportError('This module requires dgl and dgllife')


class MoleculeImageEncoder_my_v4(MoleculeImageEncoder_my_v2):
    """
    Smaller image branch for v4.

    The convolutional backbone is preserved for stability, but the head and
    embedding size are reduced so the image modality acts as a lightweight
    auxiliary signal rather than a second dominant branch.
    """

    def __init__(self,
                 output_dim: int,
                 image_hidden_list: Sequence[int] = (192,),
                 image_embeddings: int = 128,
                 image_activation: str = 'relu',
                 image_dropout_p: float = 0.12,
                 image_dropout_at_input_no_act: bool = False,
                 attention_stage: int = 3,
                 attention_memory: int = 4) -> None:
        super(MoleculeImageEncoder_my_v4,
              self).__init__(output_dim=output_dim,
                             image_hidden_list=image_hidden_list,
                             image_embeddings=image_embeddings,
                             image_activation=image_activation,
                             image_dropout_p=image_dropout_p,
                             image_dropout_at_input_no_act=
                             image_dropout_at_input_no_act,
                             attention_stage=attention_stage,
                             attention_memory=attention_memory)


class ResidualLogitFusionHead_my_v4(ResidualLogitFusionHead_my_v3):
    """
    Lower-capacity residual fusion head for v4.

    The correction path is intentionally compact to reduce the risk of
    overriding a strong graph-only baseline.
    """

    def __init__(self,
                 embedding_dim: int,
                 output_dim: int,
                 fusion_hidden_list: Sequence[int] = (128,),
                 fusion_activation: str = 'relu',
                 fusion_dropout_p: float = 0.12,
                 fusion_dropout_at_input_no_act: bool = False,
                 initial_embedding_gate_bias: float = -2.5,
                 initial_logit_gate_bias: float = -2.5) -> None:
        super(ResidualLogitFusionHead_my_v4,
              self).__init__(embedding_dim=embedding_dim,
                             output_dim=output_dim,
                             fusion_hidden_list=fusion_hidden_list,
                             fusion_activation=fusion_activation,
                             fusion_dropout_p=fusion_dropout_p,
                             fusion_dropout_at_input_no_act=
                             fusion_dropout_at_input_no_act,
                             initial_embedding_gate_bias=
                             initial_embedding_gate_bias,
                             initial_logit_gate_bias=
                             initial_logit_gate_bias)


class CustomMultiLabelAuxiliaryLoss_my_v4(object):
    """
    Loss for v4 that combines fused-output supervision with auxiliary graph
    branch supervision.
    """

    def __init__(self,
                 model: TorchModel,
                 class_imbalance_ratio: Optional[List] = None,
                 loss_aggr_type: str = 'sum',
                 device: Optional[str] = None,
                 graph_auxiliary_weight: float = 0.5):
        self.model = model
        self.loss_aggr_type = loss_aggr_type
        self.graph_auxiliary_weight = graph_auxiliary_weight
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        if class_imbalance_ratio is None:
            print(Warning("No class imbalance ratio provided!"))
            self.class_imbalance_ratio: Optional[torch.Tensor] = None
        else:
            self.class_imbalance_ratio = torch.tensor(class_imbalance_ratio,
                                                      dtype=torch.float32)
            if device is not None:
                self.class_imbalance_ratio = self.class_imbalance_ratio.to(
                    device)

        if loss_aggr_type not in ['sum', 'mean']:
            raise ValueError(f"Invalid loss aggregate type: {loss_aggr_type}")

    def _single_branch_loss(self, output: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        if len(output.shape) == 3:
            output = output.permute(0, 2, 1)

        if len(labels.shape) == len(output.shape):
            labels = labels.squeeze(-1)

        probabilities: torch.Tensor = output[:, 0, :]
        complement_probabilities: torch.Tensor = 1 - probabilities
        binary_output: torch.Tensor = torch.stack(
            [complement_probabilities, probabilities], dim=1)

        ce_loss: torch.Tensor = self.ce_loss_fn(binary_output, labels.long())

        if self.class_imbalance_ratio is None:
            if self.loss_aggr_type == 'sum':
                loss: torch.Tensor = ce_loss.sum(dim=1)
            else:
                loss = ce_loss.mean(dim=1)
        else:
            balancing_factors: torch.Tensor = torch.log(
                1 + self.class_imbalance_ratio)
            balanced_losses: torch.Tensor = torch.mul(ce_loss,
                                                      balancing_factors)
            if self.loss_aggr_type == 'sum':
                loss = balanced_losses.sum(dim=1)
            else:
                loss = balanced_losses.mean(dim=1)

        return loss.unsqueeze(-1).repeat(1, output.shape[-1])

    def __call__(self, outputs: List[torch.Tensor], labels: List[torch.Tensor],
                 weights: List[torch.Tensor]) -> torch.Tensor:
        if len(outputs) != 2 or len(labels) != 1 or len(weights) != 1:
            raise ValueError(
                "CustomMultiLabelAuxiliaryLoss_my_v4 expects two outputs (fused logits and graph logits), one label tensor, and one weight tensor"
            )

        fused_logits, graph_logits = outputs
        labels_tensor = labels[0]
        fused_losses = self._single_branch_loss(fused_logits, labels_tensor)
        graph_losses = self._single_branch_loss(graph_logits, labels_tensor)
        losses = fused_losses + self.graph_auxiliary_weight * graph_losses

        w = weights[0]
        if len(w.shape) < len(losses.shape):
            if isinstance(w, torch.Tensor):
                shape = tuple(w.shape)
            else:
                shape = w.shape
            shape = tuple(-1 if x is None else x for x in shape)
            w = w.reshape(shape + (1,) * (len(losses.shape) - len(w.shape)))

        loss = (losses * w).mean()
        if self.model.regularization_loss is not None:
            loss = loss + self.model.regularization_loss()
        return loss


class MPNNPOM_my_v4(MPNNPOM_my_v3):
    """
    Multimodal POM model for v4.

    Compared with v3, this version keeps the graph branch closer to v1, uses a
    smaller image branch, and exposes graph logits for auxiliary supervision.
    """

    def __init__(self,
                 n_tasks: int,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 mode: str = 'classification',
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 nfeat_name: str = 'x',
                 efeat_name: str = 'edge_attr',
                 readout_type: str = 'set2set',
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 image_hidden_list: Optional[Sequence[int]] = None,
                 image_embeddings: Optional[int] = None,
                 image_activation: Optional[str] = None,
                 image_dropout_p: Optional[float] = None,
                 image_dropout_at_input_no_act: Optional[bool] = None,
                 image_attention_stage: int = 3,
                 image_attention_memory: int = 4,
                 fusion_hidden_list: Optional[Sequence[int]] = None,
                 fusion_activation: Optional[str] = None,
                 fusion_dropout_p: Optional[float] = None,
                 fusion_dropout_at_input_no_act: Optional[bool] = None):
        super(MPNNPOM_my_v4, self).__init__(
            n_tasks=n_tasks,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            edge_out_feats=edge_out_feats,
            num_step_message_passing=num_step_message_passing,
            mpnn_residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type,
            mode=mode,
            number_atom_features=number_atom_features,
            number_bond_features=number_bond_features,
            n_classes=n_classes,
            nfeat_name=nfeat_name,
            efeat_name=efeat_name,
            readout_type=readout_type,
            num_step_set2set=num_step_set2set,
            num_layer_set2set=num_layer_set2set,
            ffn_hidden_list=ffn_hidden_list,
            ffn_embeddings=ffn_embeddings,
            ffn_activation=ffn_activation,
            ffn_dropout_p=ffn_dropout_p,
            ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act,
            image_hidden_list=image_hidden_list,
            image_embeddings=image_embeddings,
            image_activation=image_activation,
            image_dropout_p=image_dropout_p,
            image_dropout_at_input_no_act=image_dropout_at_input_no_act,
            image_attention_stage=image_attention_stage,
            image_attention_memory=image_attention_memory,
            fusion_hidden_list=fusion_hidden_list,
            fusion_activation=fusion_activation,
            fusion_dropout_p=fusion_dropout_p,
            fusion_dropout_at_input_no_act=fusion_dropout_at_input_no_act)

        self.mpnn = CustomMPNNGNN_my_v4(
            node_in_feats=number_atom_features,
            node_out_feats=node_out_feats,
            edge_in_feats=number_bond_features,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
            residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type,
            dropout=0.10,
            layer_norm=True,
            jk='attention')

        final_image_hidden_list = list(
            image_hidden_list) if image_hidden_list is not None else [192]
        final_image_embeddings = image_embeddings if image_embeddings is not None else 128
        final_image_activation = image_activation if image_activation is not None else ffn_activation
        final_image_dropout_p = image_dropout_p if image_dropout_p is not None else ffn_dropout_p
        final_image_dropout_at_input_no_act = (
            image_dropout_at_input_no_act
            if image_dropout_at_input_no_act is not None else
            ffn_dropout_at_input_no_act)
        final_fusion_hidden_list = list(
            fusion_hidden_list) if fusion_hidden_list is not None else [128]
        final_fusion_activation = fusion_activation if fusion_activation is not None else ffn_activation
        final_fusion_dropout_p = fusion_dropout_p if fusion_dropout_p is not None else ffn_dropout_p
        final_fusion_dropout_at_input_no_act = (
            fusion_dropout_at_input_no_act
            if fusion_dropout_at_input_no_act is not None else
            ffn_dropout_at_input_no_act)

        self.image_encoder = MoleculeImageEncoder_my_v4(
            output_dim=self.ffn_output,
            image_hidden_list=final_image_hidden_list,
            image_embeddings=final_image_embeddings,
            image_activation=final_image_activation,
            image_dropout_p=final_image_dropout_p,
            image_dropout_at_input_no_act=
            final_image_dropout_at_input_no_act,
            attention_stage=image_attention_stage,
            attention_memory=image_attention_memory)

        if final_image_embeddings != ffn_embeddings:
            self.image_to_graph_embedding = nn.Linear(final_image_embeddings,
                                                      ffn_embeddings)
        else:
            self.image_to_graph_embedding = nn.Identity()

        self.residual_fusion_head = ResidualLogitFusionHead_my_v4(
            embedding_dim=ffn_embeddings,
            output_dim=self.ffn_output,
            fusion_hidden_list=final_fusion_hidden_list,
            fusion_activation=final_fusion_activation,
            fusion_dropout_p=final_fusion_dropout_p,
            fusion_dropout_at_input_no_act=
            final_fusion_dropout_at_input_no_act)

    def forward(
        self, inputs: Union[List[Any], Tuple[Any, ...]]
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
               torch.Tensor]:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "MPNNPOM_my_v4 expects [batched_graph, batched_images] as input")

        g: DGLGraph = inputs[0]
        images: torch.Tensor = inputs[1]

        node_feats: torch.Tensor = g.ndata[self.nfeat_name]
        edge_feats: torch.Tensor = g.edata[self.efeat_name]

        node_encodings: torch.Tensor = self.mpnn(g, node_feats, edge_feats)
        molecular_encodings: torch.Tensor = self._readout(
            g, node_encodings, edge_feats)
        if self.readout_type == 'global_sum_pooling':
            molecular_encodings = F.softmax(molecular_encodings, dim=1)

        graph_embeddings, graph_logits = self.graph_ffn(molecular_encodings)
        if graph_embeddings is None:
            raise ValueError(
                "graph branch requires a non-empty hidden stack to expose embeddings")

        image_embeddings, image_logits = self.image_encoder(images)
        image_embeddings = self.image_to_graph_embedding(image_embeddings)

        fused_embeddings, out, _, _ = self.residual_fusion_head(
            graph_embeddings, image_embeddings, graph_logits, image_logits)

        if self.mode == 'classification':
            logits = self._reshape_logits(out)
            graph_loss_logits = self._reshape_logits(graph_logits)
            proba: torch.Tensor = torch.sigmoid(logits)
            if self.n_classes == 1:
                proba = proba.squeeze(-1)
            return proba, logits, fused_embeddings, graph_loss_logits
        else:
            return out


class MPNNPOMModel_my_v4(MPNNPOMModel_my_v2):
    """
    TorchModel wrapper for v4.

    v4 adds graph auxiliary supervision and uses a lower-capacity image/correction
    path while preserving the existing multimodal batching logic.
    """

    def __init__(self,
                 n_tasks: int,
                 class_imbalance_ratio: Optional[List] = None,
                 loss_aggr_type: str = 'sum',
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 batch_size: int = 100,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 mode: str = 'regression',
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 readout_type: str = 'set2set',
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 weight_decay: float = 1e-5,
                 self_loop: bool = False,
                 optimizer_name: str = 'adam',
                 device_name: Optional[str] = None,
                 image_hidden_list: Optional[Sequence[int]] = None,
                 image_embeddings: Optional[int] = None,
                 image_activation: Optional[str] = None,
                 image_dropout_p: Optional[float] = None,
                 image_dropout_at_input_no_act: Optional[bool] = None,
                 image_attention_stage: int = 3,
                 image_attention_memory: int = 4,
                 fusion_hidden_list: Optional[Sequence[int]] = None,
                 fusion_activation: Optional[str] = None,
                 fusion_dropout_p: Optional[float] = None,
                 fusion_dropout_at_input_no_act: Optional[bool] = None,
                 image_cache_dir: Optional[Union[str, Path]] = None,
                 export_image_size: int = 256,
                 input_image_size: int = 128,
                 graph_auxiliary_weight: float = 0.5,
                 **kwargs):
        model: nn.Module = MPNNPOM_my_v4(
            n_tasks=n_tasks,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            edge_out_feats=edge_out_feats,
            num_step_message_passing=num_step_message_passing,
            mpnn_residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type,
            mode=mode,
            number_atom_features=number_atom_features,
            number_bond_features=number_bond_features,
            n_classes=n_classes,
            readout_type=readout_type,
            num_step_set2set=num_step_set2set,
            num_layer_set2set=num_layer_set2set,
            ffn_hidden_list=ffn_hidden_list,
            ffn_embeddings=ffn_embeddings,
            ffn_activation=ffn_activation,
            ffn_dropout_p=ffn_dropout_p,
            ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act,
            image_hidden_list=image_hidden_list,
            image_embeddings=image_embeddings,
            image_activation=image_activation,
            image_dropout_p=image_dropout_p,
            image_dropout_at_input_no_act=image_dropout_at_input_no_act,
            image_attention_stage=image_attention_stage,
            image_attention_memory=image_attention_memory,
            fusion_hidden_list=fusion_hidden_list,
            fusion_activation=fusion_activation,
            fusion_dropout_p=fusion_dropout_p,
            fusion_dropout_at_input_no_act=fusion_dropout_at_input_no_act)

        if class_imbalance_ratio and (len(class_imbalance_ratio) != n_tasks):
            raise Exception("size of class_imbalance_ratio \
                            should be equal to n_tasks")

        if mode == 'regression':
            loss: Union[Loss, Callable] = L2Loss()
            output_types: List = ['prediction']
        else:
            loss = CustomMultiLabelAuxiliaryLoss_my_v4(
                model=self,
                class_imbalance_ratio=class_imbalance_ratio,
                loss_aggr_type=loss_aggr_type,
                device=device_name,
                graph_auxiliary_weight=graph_auxiliary_weight)
            output_types = ['prediction', 'loss', 'embedding', 'loss']

        optimizer: Optimizer = get_optimizer(optimizer_name)
        optimizer.learning_rate = learning_rate
        if device_name is not None:
            device: Optional[torch.device] = torch.device(device_name)
        else:
            device = None
        TorchModel.__init__(self,
                            model,
                            loss=loss,
                            output_types=output_types,
                            optimizer=optimizer,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            device=device,
                            **kwargs)

        self.weight_decay: float = weight_decay
        self._self_loop: bool = self_loop
        self.regularization_loss: Callable = self._regularization_loss
        cache_root = image_cache_dir
        if cache_root is None:
            cache_root = Path(self.model_dir) / "smiles_png_cache_my_v4"
        self.image_cache = SmilesImageCache_my_v2(
            cache_dir=cache_root,
            export_image_size=export_image_size,
            input_image_size=input_image_size)


