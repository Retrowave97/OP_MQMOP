from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from deepchem.models.losses import L2Loss, Loss
from deepchem.models.optimizers import (AdaGrad, Adam, AdamW,
                                        GradientDescent, KFAC,
                                        LearningRateSchedule, Optimizer,
                                        RMSProp, SparseAdam)
from deepchem.models.torch_models.torch_model import TorchModel

try:
    import dgl
    from dgl import DGLGraph
    from dgl.nn.pytorch import NNConv, Set2Set
except (ImportError, ModuleNotFoundError):
    raise ImportError("This module requires dgl and dgllife")


INVALID_FILENAME_CHARS_MQMOP = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
IMAGE_FILENAME_SMILES_PREFIX_LENGTH_MQMOP = 8
IMAGE_CACHE_ID_SEPARATOR_MQMOP = "\t"
PIL_BILINEAR_MQMOP = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def sanitize_smiles_for_filename_mqmop(smiles: str,
                                       max_length: int = 120) -> str:
    sanitized = INVALID_FILENAME_CHARS_MQMOP.sub("_", str(smiles).strip())
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = sanitized.rstrip(". ")
    if not sanitized:
        sanitized = "empty_smiles"
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def build_csv_row_smiles_png_filename_mqmop(row_number: int,
                                            smiles: str) -> str:
    smiles_prefix = sanitize_smiles_for_filename_mqmop(
        smiles, max_length=IMAGE_FILENAME_SMILES_PREFIX_LENGTH_MQMOP)
    return f"{int(row_number):05d}_{smiles_prefix}.png"


def parse_csv_row_smiles_image_id_mqmop(
        image_id: Any) -> Tuple[Optional[int], str]:
    text = str(image_id)
    row_number_text, separator, smiles = text.partition(
        IMAGE_CACHE_ID_SEPARATOR_MQMOP)
    if not separator:
        return None, text
    try:
        row_number = int(row_number_text)
    except ValueError:
        row_number = None
    return row_number, smiles


def smiles_to_png_mqmop(smiles: str, output_path: Path,
                        image_size: int) -> Path:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    image = Draw.MolToImage(mol, size=(image_size, image_size), kekulize=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def get_mqmop_optimizer(optimizer_name: str = "adam") -> Optimizer:
    optimizer_name = str(optimizer_name).lower()
    if optimizer_name == "adam":
        return Adam()
    if optimizer_name == "adagrad":
        return AdaGrad()
    if optimizer_name == "adamw":
        return AdamW()
    if optimizer_name == "sparseadam":
        return SparseAdam()
    if optimizer_name == "rmsprop":
        return RMSProp()
    if optimizer_name == "sgd":
        return GradientDescent()
    if optimizer_name == "kfac":
        return KFAC()
    return Adam()


class MQMOPPositionwiseFeedForward(nn.Module):
    def __init__(self,
                 d_input: int = 1024,
                 d_hidden_list: List[int] = [1024],
                 d_output: int = 1024,
                 activation: str = "leakyrelu",
                 dropout_p: float = 0.0,
                 dropout_at_input_no_act: bool = False,
                 batch_norm: bool = True):
        super(MQMOPPositionwiseFeedForward, self).__init__()

        self.dropout_at_input_no_act = dropout_at_input_no_act
        self.batch_norm = batch_norm

        if activation == "relu":
            self.activation: Callable[[Any], Any] = nn.ReLU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "linear":
            self.activation = lambda x: x
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        d_output = d_output if d_output != 0 else d_input
        self.n_layers = len(d_hidden_list) + 1

        if self.n_layers == 1:
            linears: List[nn.Module] = [nn.Linear(d_input, d_output)]
        else:
            linears = [nn.Linear(d_input, d_hidden_list[0])]
            for idx in range(1, len(d_hidden_list)):
                linears.append(
                    nn.Linear(d_hidden_list[idx - 1], d_hidden_list[idx]))
            linears.append(nn.Linear(d_hidden_list[-1], d_output))

        self.linears = nn.ModuleList(linears)
        dropout_layer = nn.Dropout(dropout_p)
        self.dropout_layers = nn.ModuleList(
            [dropout_layer for _ in range(self.n_layers)])

        if batch_norm:
            batchnorms = [
                nn.BatchNorm1d(d_hidden_list[idx])
                for idx in range(len(d_hidden_list))
            ]
            self.batchnorms = nn.ModuleList(batchnorms)
        else:
            self.batchnorms = None

    def forward(self, x: torch.Tensor) -> List[Optional[torch.Tensor]]:
        if self.n_layers == 1:
            if self.dropout_at_input_no_act:
                return [None, self.linears[0](self.dropout_layers[0](x))]
            return [
                None,
                self.dropout_layers[0](self.activation(self.linears[0](x)))
            ]

        if self.dropout_at_input_no_act:
            x = self.dropout_layers[-1](x)

        if self.batch_norm:
            assert self.batchnorms is not None
            for idx in range(self.n_layers - 2):
                x = self.dropout_layers[idx](self.activation(
                    self.batchnorms[idx](self.linears[idx](x))))
            embeddings = self.linears[self.n_layers - 2](x)
            x = self.dropout_layers[self.n_layers - 2](self.activation(
                self.batchnorms[self.n_layers - 2](embeddings)))
        else:
            for idx in range(self.n_layers - 2):
                x = self.dropout_layers[idx](self.activation(
                    self.linears[idx](x)))
            embeddings = self.linears[self.n_layers - 2](x)
            x = self.dropout_layers[self.n_layers - 2](
                self.activation(embeddings))

        output = self.linears[-1](x)
        return [embeddings, output]


class CustomMQMOPGNN(nn.Module):
    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = "sum",
                 dropout: float = 0.10,
                 layer_norm: bool = True,
                 jk: str = "attention"):
        super(CustomMQMOPGNN, self).__init__()

        if jk not in ["last", "mean", "max", "attention"]:
            raise ValueError(
                "jk must be one of ['last', 'mean', 'max', 'attention']")

        self.node_out_feats = node_out_feats
        self.num_step_message_passing = num_step_message_passing
        self.layer_norm = layer_norm
        self.jk = jk

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats), nn.ReLU())
        self.dropout = nn.Dropout(dropout)

        self.gnn_layers = nn.ModuleList()
        self.update_grus = nn.ModuleList()
        self.norms = nn.ModuleList() if layer_norm else None

        for _ in range(num_step_message_passing):
            edge_network = nn.Sequential(
                nn.Linear(edge_in_feats, edge_hidden_feats),
                nn.ReLU(),
                nn.Linear(edge_hidden_feats, edge_hidden_feats),
                nn.ReLU(),
                nn.Linear(edge_hidden_feats,
                          node_out_feats * node_out_feats))
            self.gnn_layers.append(
                NNConv(in_feats=node_out_feats,
                       out_feats=node_out_feats,
                       edge_func=edge_network,
                       aggregator_type=message_aggregator_type,
                       residual=residual))
            self.update_grus.append(nn.GRUCell(node_out_feats, node_out_feats))
            if layer_norm:
                assert self.norms is not None
                self.norms.append(nn.LayerNorm(node_out_feats))

        if jk == "attention":
            self.step_logits = nn.Parameter(
                torch.zeros(num_step_message_passing, dtype=torch.float32))

    def reset_parameters(self) -> None:
        self.project_node_feats[0].reset_parameters()
        for gnn_layer in self.gnn_layers:
            gnn_layer.reset_parameters()
            for layer in gnn_layer.edge_func:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
        for gru in self.update_grus:
            gru.reset_parameters()
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()
        if hasattr(self, "step_logits"):
            nn.init.zeros_(self.step_logits)

    def _combine_step_outputs(self,
                              step_outputs: List[torch.Tensor]) -> torch.Tensor:
        if len(step_outputs) == 0:
            raise ValueError("step_outputs should contain at least one tensor")
        if self.jk == "last":
            return step_outputs[-1]

        stacked_outputs = torch.stack(step_outputs, dim=0)
        if self.jk == "mean":
            return stacked_outputs.mean(dim=0)
        if self.jk == "max":
            return stacked_outputs.max(dim=0).values

        step_weights = torch.softmax(self.step_logits[:len(step_outputs)], dim=0)
        view_shape = (len(step_outputs),) + (1,) * (stacked_outputs.dim() - 1)
        return torch.sum(stacked_outputs * step_weights.view(view_shape), dim=0)

    def forward(self, g: DGLGraph, node_feats: torch.Tensor,
                edge_feats: torch.Tensor) -> torch.Tensor:
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats
        step_outputs: List[torch.Tensor] = []

        for step_idx, (gnn_layer, gru) in enumerate(
                zip(self.gnn_layers, self.update_grus)):
            messages = F.relu(gnn_layer(g, node_feats, edge_feats))
            messages = self.dropout(messages)
            updated_feats = gru(messages, hidden_feats)
            if self.norms is not None:
                updated_feats = self.norms[step_idx](updated_feats)
            updated_feats = self.dropout(updated_feats)
            node_feats = updated_feats
            hidden_feats = updated_feats
            step_outputs.append(updated_feats)

        return self._combine_step_outputs(step_outputs)


class MQMOPImageCache:
    def __init__(self,
                 cache_dir: Union[str, Path],
                 export_image_size: int = 256,
                 input_image_size: int = 128) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.export_image_size = export_image_size
        self.input_image_size = input_image_size

    def build_png_path(self,
                       smiles: str,
                       csv_row_number: Optional[int] = None) -> Path:
        smiles = str(smiles)
        if csv_row_number is None:
            digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:12]
            filename = (
                f"{digest}_"
                f"{sanitize_smiles_for_filename_mqmop(smiles)}.png")
        else:
            filename = build_csv_row_smiles_png_filename_mqmop(
                csv_row_number, smiles)
        return self.cache_dir / filename

    def load_tensor(self,
                    smiles: str,
                    csv_row_number: Optional[int] = None) -> torch.Tensor:
        png_path = self.build_png_path(smiles, csv_row_number)
        if not png_path.exists():
            smiles_to_png_mqmop(smiles, png_path, self.export_image_size)
        image = Image.open(png_path).convert("RGB")
        if image.size != (self.input_image_size, self.input_image_size):
            image = image.resize((self.input_image_size, self.input_image_size),
                                 PIL_BILINEAR_MQMOP)
        return torchvision.transforms.functional.to_tensor(image).float()


class MQMOPExternalAttention(nn.Module):
    def __init__(self, d_model: int, s: int = 64) -> None:
        super(MQMOPExternalAttention, self).__init__()
        self.mk = nn.Linear(d_model, s, bias=False)
        self.mv = nn.Linear(s, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = queries.shape
        flattened = queries.permute(0, 1, 3, 2).contiguous().view(-1, height)
        attention = self.mk(flattened)
        attention = self.softmax(attention)
        attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-12)
        out = self.mv(attention)
        return out.view(batch_size, channels, width, height).permute(
            0, 1, 3, 2).contiguous()


class MQMOPBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(MQMOPBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class MQMOPImageEncoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 image_hidden_list: Sequence[int] = (192,),
                 image_embeddings: int = 128,
                 image_activation: str = "relu",
                 image_dropout_p: float = 0.12,
                 image_dropout_at_input_no_act: bool = False,
                 attention_stage: int = 3,
                 attention_memory: int = 4) -> None:
        super(MQMOPImageEncoder, self).__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(MQMOPBasicBlock, 32, 3, stride=1)
        self.layer2 = self._make_layer(MQMOPBasicBlock, 64, 4, stride=2)
        self.layer3 = self._make_layer(MQMOPBasicBlock, 128, 6, stride=2)
        self.layer4 = self._make_layer(MQMOPBasicBlock, 256, 3, stride=2)
        self.attention_stage = attention_stage
        self.attention_modules = nn.ModuleDict({
            "1": MQMOPExternalAttention(d_model=128, s=attention_memory),
            "2": MQMOPExternalAttention(d_model=64, s=attention_memory),
            "3": MQMOPExternalAttention(d_model=32, s=attention_memory),
            "4": MQMOPExternalAttention(d_model=16, s=attention_memory),
        })
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        image_hidden_dims = list(image_hidden_list) + [image_embeddings]
        self.head = MQMOPPositionwiseFeedForward(
            d_input=256 * 16,
            d_hidden_list=image_hidden_dims,
            d_output=output_dim,
            activation=image_activation,
            dropout_p=image_dropout_p,
            dropout_at_input_no_act=image_dropout_at_input_no_act)
        self.embedding_dim = image_embeddings

    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for current_stride in strides:
            layers.append(block(self.in_planes, planes, current_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.attention_stage == 1:
            out = self.attention_modules["1"](out)
        out = self.layer2(out)
        if self.attention_stage == 2:
            out = self.attention_modules["2"](out)
        out = self.layer3(out)
        if self.attention_stage == 3:
            out = self.attention_modules["3"](out)
        out = self.layer4(out)
        if self.attention_stage == 4:
            out = self.attention_modules["4"](out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        image_embeddings, logits = self.head(out)
        if image_embeddings is None:
            raise ValueError(
                "image branch requires a non-empty hidden stack to expose embeddings"
            )
        return image_embeddings, logits


class MQMOPModalityFusionHead(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 output_dim: int,
                 fusion_hidden_list: Sequence[int] = (392, 392),
                 fusion_activation: str = "relu",
                 fusion_dropout_p: float = 0.12,
                 fusion_dropout_at_input_no_act: bool = False) -> None:
        super(MQMOPModalityFusionHead, self).__init__()
        interaction_dim = embedding_dim * 4
        self.fusion_projection = MQMOPPositionwiseFeedForward(
            d_input=interaction_dim,
            d_hidden_list=list(fusion_hidden_list),
            d_output=embedding_dim,
            activation=fusion_activation,
            dropout_p=fusion_dropout_p,
            dropout_at_input_no_act=fusion_dropout_at_input_no_act)
        self.modality_gate = nn.Linear(interaction_dim, embedding_dim)
        self.output_dropout = (nn.Dropout(fusion_dropout_p)
                               if fusion_dropout_p > 0 else nn.Identity())
        self.output_layer = nn.Linear(embedding_dim, output_dim)

    def forward(
        self, graph_embeddings: torch.Tensor, image_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        interaction = torch.cat([
            graph_embeddings, image_embeddings,
            torch.abs(graph_embeddings - image_embeddings),
            graph_embeddings * image_embeddings
        ],
                                dim=1)
        _, fusion_delta = self.fusion_projection(interaction)
        gate = torch.sigmoid(self.modality_gate(interaction))
        fused_embeddings = gate * graph_embeddings + (1 - gate
                                                      ) * image_embeddings
        fused_embeddings = fused_embeddings + fusion_delta
        fused_embeddings = self.output_dropout(fused_embeddings)
        logits = self.output_layer(fused_embeddings)
        return fused_embeddings, logits, gate


class MQMOPResidualFusionHead(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 output_dim: int,
                 fusion_hidden_list: Sequence[int] = (128,),
                 fusion_activation: str = "relu",
                 fusion_dropout_p: float = 0.12,
                 fusion_dropout_at_input_no_act: bool = False,
                 initial_embedding_gate_bias: float = -2.5,
                 initial_logit_gate_bias: float = -2.5) -> None:
        super(MQMOPResidualFusionHead, self).__init__()
        interaction_dim = embedding_dim * 4
        gate_context_dim = interaction_dim + output_dim
        self.correction_encoder = MQMOPPositionwiseFeedForward(
            d_input=interaction_dim,
            d_hidden_list=list(fusion_hidden_list),
            d_output=embedding_dim,
            activation=fusion_activation,
            dropout_p=fusion_dropout_p,
            dropout_at_input_no_act=fusion_dropout_at_input_no_act)
        self.embedding_gate = nn.Linear(interaction_dim, embedding_dim)
        self.logit_gate = nn.Linear(gate_context_dim, output_dim)
        self.output_dropout = (nn.Dropout(fusion_dropout_p)
                               if fusion_dropout_p > 0 else nn.Identity())
        self.output_layer = nn.Linear(embedding_dim, output_dim)
        self._init_residual_layers(initial_embedding_gate_bias,
                                   initial_logit_gate_bias)

    def _init_residual_layers(self, embedding_gate_bias: float,
                              logit_gate_bias: float) -> None:
        nn.init.zeros_(self.embedding_gate.weight)
        nn.init.constant_(self.embedding_gate.bias, embedding_gate_bias)
        nn.init.zeros_(self.logit_gate.weight)
        nn.init.constant_(self.logit_gate.bias, logit_gate_bias)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(
        self, graph_embeddings: torch.Tensor, image_embeddings: torch.Tensor,
        graph_logits: torch.Tensor, image_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        interaction = torch.cat([
            graph_embeddings, image_embeddings,
            torch.abs(graph_embeddings - image_embeddings),
            graph_embeddings * image_embeddings
        ],
                                dim=1)
        logit_delta = image_logits - graph_logits
        gate_context = torch.cat([interaction, logit_delta], dim=1)

        _, correction_embeddings = self.correction_encoder(interaction)
        embedding_gate = torch.sigmoid(self.embedding_gate(interaction))
        residual_embeddings = embedding_gate * correction_embeddings
        residual_embeddings = self.output_dropout(residual_embeddings)

        residual_logits = self.output_layer(residual_embeddings)
        logit_gate = torch.sigmoid(self.logit_gate(gate_context))
        residual_logits = logit_gate * residual_logits

        fused_embeddings = graph_embeddings + residual_embeddings
        final_logits = graph_logits + residual_logits
        return fused_embeddings, final_logits, residual_logits, logit_gate


class MQMOPAuxiliaryLoss(object):
    def __init__(self,
                 model: TorchModel,
                 class_imbalance_ratio: Optional[List] = None,
                 loss_aggr_type: str = "sum",
                 device: Optional[str] = None,
                 graph_auxiliary_weight: float = 0.5):
        self.model = model
        self.loss_aggr_type = loss_aggr_type
        self.graph_auxiliary_weight = graph_auxiliary_weight
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        if class_imbalance_ratio is None:
            self.class_imbalance_ratio: Optional[torch.Tensor] = None
        else:
            self.class_imbalance_ratio = torch.tensor(class_imbalance_ratio,
                                                      dtype=torch.float32)
            if device is not None:
                self.class_imbalance_ratio = self.class_imbalance_ratio.to(
                    device)

        if loss_aggr_type not in ["sum", "mean"]:
            raise ValueError(
                f"Invalid loss aggregate type: {loss_aggr_type}")

    def _single_branch_loss(self, output: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        if len(output.shape) == 3:
            output = output.permute(0, 2, 1)

        if len(labels.shape) == len(output.shape):
            labels = labels.squeeze(-1)

        probabilities = output[:, 0, :]
        complement_probabilities = 1 - probabilities
        binary_output = torch.stack([complement_probabilities, probabilities],
                                    dim=1)
        ce_loss = self.ce_loss_fn(binary_output, labels.long())

        if self.class_imbalance_ratio is None:
            if self.loss_aggr_type == "sum":
                loss = ce_loss.sum(dim=1)
            else:
                loss = ce_loss.mean(dim=1)
        else:
            balancing_factors = torch.log(1 + self.class_imbalance_ratio)
            balanced_losses = torch.mul(ce_loss, balancing_factors)
            if self.loss_aggr_type == "sum":
                loss = balanced_losses.sum(dim=1)
            else:
                loss = balanced_losses.mean(dim=1)

        return loss.unsqueeze(-1).repeat(1, output.shape[-1])

    def __call__(self, outputs: List[torch.Tensor], labels: List[torch.Tensor],
                 weights: List[torch.Tensor]) -> torch.Tensor:
        if len(outputs) != 2 or len(labels) != 1 or len(weights) != 1:
            raise ValueError(
                "MQMOPAuxiliaryLoss expects two outputs (fused logits and graph logits), one label tensor, and one weight tensor"
            )

        fused_logits, graph_logits = outputs
        labels_tensor = labels[0]
        fused_losses = self._single_branch_loss(fused_logits, labels_tensor)
        graph_losses = self._single_branch_loss(graph_logits, labels_tensor)
        losses = fused_losses + self.graph_auxiliary_weight * graph_losses

        w = weights[0]
        if len(w.shape) < len(losses.shape):
            shape = tuple(w.shape)
            shape = tuple(-1 if x is None else x for x in shape)
            w = w.reshape(shape + (1,) * (len(losses.shape) - len(w.shape)))

        loss = (losses * w).mean()
        if self.model.regularization_loss is not None:
            loss = loss + self.model.regularization_loss()
        return loss


class _MQMOPStageV2(nn.Module):
    def __init__(self,
                 n_tasks: int,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = "sum",
                 mode: str = "classification",
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 nfeat_name: str = "x",
                 efeat_name: str = "edge_attr",
                 readout_type: str = "set2set",
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List[int] = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = "relu",
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 image_hidden_list: Optional[Sequence[int]] = None,
                 image_embeddings: Optional[int] = None,
                 image_activation: Optional[str] = None,
                 image_dropout_p: Optional[float] = None,
                 image_dropout_at_input_no_act: Optional[bool] = None,
                 image_attention_stage: int = 3,
                 image_attention_memory: int = 8,
                 fusion_hidden_list: Optional[Sequence[int]] = None,
                 fusion_activation: Optional[str] = None,
                 fusion_dropout_p: Optional[float] = None,
                 fusion_dropout_at_input_no_act: Optional[bool] = None):
        if mode not in ["classification", "regression"]:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        super(_MQMOPStageV2, self).__init__()

        self.n_tasks = n_tasks
        self.mode = mode
        self.n_classes = n_classes
        self.nfeat_name = nfeat_name
        self.efeat_name = efeat_name
        self.readout_type = readout_type

        if mode == "classification":
            self.ffn_output = n_tasks * n_classes
        else:
            self.ffn_output = n_tasks

        self.mpnn = CustomMQMOPGNN(
            node_in_feats=number_atom_features,
            node_out_feats=node_out_feats,
            edge_in_feats=number_bond_features,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
            residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type,
            dropout=0.10,
            layer_norm=True,
            jk="attention")

        self.project_edge_feats = nn.Sequential(
            nn.Linear(number_bond_features, edge_out_feats), nn.ReLU())

        if self.readout_type == "set2set":
            self.readout_set2set = Set2Set(input_dim=node_out_feats +
                                           edge_out_feats,
                                           n_iters=num_step_set2set,
                                           n_layers=num_layer_set2set)
            graph_ffn_input = 2 * (node_out_feats + edge_out_feats)
        elif self.readout_type == "global_sum_pooling":
            graph_ffn_input = node_out_feats + edge_out_feats
        else:
            raise ValueError("readout_type invalid")

        graph_hidden_dims = list(ffn_hidden_list) + [ffn_embeddings]
        self.graph_ffn = MQMOPPositionwiseFeedForward(
            d_input=graph_ffn_input,
            d_hidden_list=graph_hidden_dims,
            d_output=self.ffn_output,
            activation=ffn_activation,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act)

        image_hidden_list = list(
            image_hidden_list) if image_hidden_list is not None else list(
                ffn_hidden_list)
        image_embeddings = image_embeddings if image_embeddings is not None else ffn_embeddings
        image_activation = image_activation if image_activation is not None else ffn_activation
        image_dropout_p = image_dropout_p if image_dropout_p is not None else ffn_dropout_p
        image_dropout_at_input_no_act = (
            image_dropout_at_input_no_act
            if image_dropout_at_input_no_act is not None else
            ffn_dropout_at_input_no_act)
        fusion_hidden_list = list(
            fusion_hidden_list) if fusion_hidden_list is not None else list(
                ffn_hidden_list)
        fusion_activation = fusion_activation if fusion_activation is not None else ffn_activation
        fusion_dropout_p = fusion_dropout_p if fusion_dropout_p is not None else ffn_dropout_p
        fusion_dropout_at_input_no_act = (
            fusion_dropout_at_input_no_act
            if fusion_dropout_at_input_no_act is not None else
            ffn_dropout_at_input_no_act)

        self.image_encoder = MQMOPImageEncoder(
            output_dim=self.ffn_output,
            image_hidden_list=image_hidden_list,
            image_embeddings=image_embeddings,
            image_activation=image_activation,
            image_dropout_p=image_dropout_p,
            image_dropout_at_input_no_act=image_dropout_at_input_no_act,
            attention_stage=image_attention_stage,
            attention_memory=image_attention_memory)

        if image_embeddings != ffn_embeddings:
            self.image_to_graph_embedding = nn.Linear(image_embeddings,
                                                      ffn_embeddings)
        else:
            self.image_to_graph_embedding = nn.Identity()

        self.fusion_head = MQMOPModalityFusionHead(
            embedding_dim=ffn_embeddings,
            output_dim=self.ffn_output,
            fusion_hidden_list=fusion_hidden_list,
            fusion_activation=fusion_activation,
            fusion_dropout_p=fusion_dropout_p,
            fusion_dropout_at_input_no_act=fusion_dropout_at_input_no_act)

    def _readout(self, g: DGLGraph, node_encodings: torch.Tensor,
                 edge_feats: torch.Tensor) -> torch.Tensor:
        g.ndata["node_emb"] = node_encodings
        g.edata["edge_emb"] = self.project_edge_feats(edge_feats)

        def message_func(edges) -> Dict[str, torch.Tensor]:
            src_msg = torch.cat((edges.src["node_emb"], edges.data["edge_emb"]),
                                dim=1)
            return {"src_msg": src_msg}

        def reduce_func(nodes) -> Dict[str, torch.Tensor]:
            src_msg_sum = torch.sum(nodes.mailbox["src_msg"], dim=1)
            return {"src_msg_sum": src_msg_sum}

        g.send_and_recv(g.edges(),
                        message_func=message_func,
                        reduce_func=reduce_func)

        if self.readout_type == "set2set":
            return self.readout_set2set(g, g.ndata["src_msg_sum"])
        return dgl.sum_nodes(g, "src_msg_sum")

    def _reshape_logits(self, out: torch.Tensor) -> torch.Tensor:
        if self.n_tasks == 1:
            return out.view(-1, self.n_classes)
        return out.view(-1, self.n_tasks, self.n_classes)

    def forward(
        self, inputs: Union[List[Any], Tuple[Any, ...]]
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "_MQMOPStageV2 expects [batched_graph, batched_images] as input"
            )

        g = inputs[0]
        images = inputs[1]

        node_feats = g.ndata[self.nfeat_name]
        edge_feats = g.edata[self.efeat_name]

        node_encodings = self.mpnn(g, node_feats, edge_feats)
        molecular_encodings = self._readout(g, node_encodings, edge_feats)
        if self.readout_type == "global_sum_pooling":
            molecular_encodings = F.softmax(molecular_encodings, dim=1)

        graph_embeddings, _ = self.graph_ffn(molecular_encodings)
        if graph_embeddings is None:
            raise ValueError(
                "graph branch requires a non-empty hidden stack to expose embeddings"
            )

        image_embeddings, _ = self.image_encoder(images)
        image_embeddings = self.image_to_graph_embedding(image_embeddings)

        fused_embeddings, out, _ = self.fusion_head(graph_embeddings,
                                                    image_embeddings)

        if self.mode == "classification":
            logits = self._reshape_logits(out)
            proba = torch.sigmoid(logits)
            if self.n_classes == 1:
                proba = proba.squeeze(-1)
            return proba, logits, fused_embeddings
        return out


class _MQMOPStageV3(_MQMOPStageV2):
    def __init__(self,
                 n_tasks: int,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = "sum",
                 mode: str = "classification",
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 nfeat_name: str = "x",
                 efeat_name: str = "edge_attr",
                 readout_type: str = "set2set",
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List[int] = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = "relu",
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 image_hidden_list: Optional[Sequence[int]] = None,
                 image_embeddings: Optional[int] = None,
                 image_activation: Optional[str] = None,
                 image_dropout_p: Optional[float] = None,
                 image_dropout_at_input_no_act: Optional[bool] = None,
                 image_attention_stage: int = 3,
                 image_attention_memory: int = 8,
                 fusion_hidden_list: Optional[Sequence[int]] = None,
                 fusion_activation: Optional[str] = None,
                 fusion_dropout_p: Optional[float] = None,
                 fusion_dropout_at_input_no_act: Optional[bool] = None):
        final_fusion_hidden_list = list(
            fusion_hidden_list) if fusion_hidden_list is not None else list(
                ffn_hidden_list)
        final_fusion_activation = fusion_activation if fusion_activation is not None else ffn_activation
        final_fusion_dropout_p = fusion_dropout_p if fusion_dropout_p is not None else ffn_dropout_p
        final_fusion_dropout_at_input_no_act = (
            fusion_dropout_at_input_no_act
            if fusion_dropout_at_input_no_act is not None else
            ffn_dropout_at_input_no_act)

        super(_MQMOPStageV3, self).__init__(
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

        self.mpnn = CustomMQMOPGNN(
            node_in_feats=number_atom_features,
            node_out_feats=node_out_feats,
            edge_in_feats=number_bond_features,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
            residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type,
            dropout=0.10,
            layer_norm=True,
            jk="attention")
        del self.fusion_head
        self.residual_fusion_head = MQMOPResidualFusionHead(
            embedding_dim=ffn_embeddings,
            output_dim=self.ffn_output,
            fusion_hidden_list=final_fusion_hidden_list,
            fusion_activation=final_fusion_activation,
            fusion_dropout_p=final_fusion_dropout_p,
            fusion_dropout_at_input_no_act=
            final_fusion_dropout_at_input_no_act,
            initial_embedding_gate_bias=-2.0,
            initial_logit_gate_bias=-2.0)

    def forward(
        self, inputs: Union[List[Any], Tuple[Any, ...]]
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "_MQMOPStageV3 expects [batched_graph, batched_images] as input"
            )

        g = inputs[0]
        images = inputs[1]

        node_feats = g.ndata[self.nfeat_name]
        edge_feats = g.edata[self.efeat_name]

        node_encodings = self.mpnn(g, node_feats, edge_feats)
        molecular_encodings = self._readout(g, node_encodings, edge_feats)
        if self.readout_type == "global_sum_pooling":
            molecular_encodings = F.softmax(molecular_encodings, dim=1)

        graph_embeddings, graph_logits = self.graph_ffn(molecular_encodings)
        if graph_embeddings is None:
            raise ValueError(
                "graph branch requires a non-empty hidden stack to expose embeddings"
            )

        image_embeddings, image_logits = self.image_encoder(images)
        image_embeddings = self.image_to_graph_embedding(image_embeddings)

        fused_embeddings, out, _, _ = self.residual_fusion_head(
            graph_embeddings, image_embeddings, graph_logits, image_logits)

        if self.mode == "classification":
            logits = self._reshape_logits(out)
            proba = torch.sigmoid(logits)
            if self.n_classes == 1:
                proba = proba.squeeze(-1)
            return proba, logits, fused_embeddings
        return out


class MQMOP(_MQMOPStageV3):
    def __init__(self,
                 n_tasks: int,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = "sum",
                 mode: str = "classification",
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 nfeat_name: str = "x",
                 efeat_name: str = "edge_attr",
                 readout_type: str = "set2set",
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List[int] = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = "relu",
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
        super(MQMOP, self).__init__(
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

        self.mpnn = CustomMQMOPGNN(
            node_in_feats=number_atom_features,
            node_out_feats=node_out_feats,
            edge_in_feats=number_bond_features,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
            residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type,
            dropout=0.10,
            layer_norm=True,
            jk="attention")

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

        self.image_encoder = MQMOPImageEncoder(
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

        self.residual_fusion_head = MQMOPResidualFusionHead(
            embedding_dim=ffn_embeddings,
            output_dim=self.ffn_output,
            fusion_hidden_list=final_fusion_hidden_list,
            fusion_activation=final_fusion_activation,
            fusion_dropout_p=final_fusion_dropout_p,
            fusion_dropout_at_input_no_act=
            final_fusion_dropout_at_input_no_act,
            initial_embedding_gate_bias=-2.5,
            initial_logit_gate_bias=-2.5)

    def forward(
        self, inputs: Union[List[Any], Tuple[Any, ...]]
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
               torch.Tensor]:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "MQMOP expects [batched_graph, batched_images] as input")

        g = inputs[0]
        images = inputs[1]

        node_feats = g.ndata[self.nfeat_name]
        edge_feats = g.edata[self.efeat_name]

        node_encodings = self.mpnn(g, node_feats, edge_feats)
        molecular_encodings = self._readout(g, node_encodings, edge_feats)
        if self.readout_type == "global_sum_pooling":
            molecular_encodings = F.softmax(molecular_encodings, dim=1)

        graph_embeddings, graph_logits = self.graph_ffn(molecular_encodings)
        if graph_embeddings is None:
            raise ValueError(
                "graph branch requires a non-empty hidden stack to expose embeddings"
            )

        image_embeddings, image_logits = self.image_encoder(images)
        image_embeddings = self.image_to_graph_embedding(image_embeddings)

        fused_embeddings, out, _, _ = self.residual_fusion_head(
            graph_embeddings, image_embeddings, graph_logits, image_logits)

        if self.mode == "classification":
            logits = self._reshape_logits(out)
            graph_loss_logits = self._reshape_logits(graph_logits)
            proba = torch.sigmoid(logits)
            if self.n_classes == 1:
                proba = proba.squeeze(-1)
            return proba, logits, fused_embeddings, graph_loss_logits
        return out


class MQMOPModel(TorchModel):
    def __init__(self,
                 n_tasks: int,
                 class_imbalance_ratio: Optional[List] = None,
                 loss_aggr_type: str = "sum",
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 batch_size: int = 100,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = "sum",
                 mode: str = "regression",
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 readout_type: str = "set2set",
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List[int] = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = "relu",
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 weight_decay: float = 1e-5,
                 self_loop: bool = False,
                 optimizer_name: str = "adam",
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
        model = MQMOP(
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
            raise ValueError(
                "size of class_imbalance_ratio should be equal to n_tasks")

        if mode == "regression":
            loss: Union[Loss, Callable] = L2Loss()
            output_types: List[str] = ["prediction"]
        else:
            loss = MQMOPAuxiliaryLoss(model=self,
                                      class_imbalance_ratio=
                                      class_imbalance_ratio,
                                      loss_aggr_type=loss_aggr_type,
                                      device=device_name,
                                      graph_auxiliary_weight=
                                      graph_auxiliary_weight)
            output_types = ["prediction", "loss", "embedding", "loss"]

        optimizer = get_mqmop_optimizer(optimizer_name)
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

        self.weight_decay = weight_decay
        self._self_loop = self_loop
        self.regularization_loss: Callable[
            [], torch.Tensor] = self._regularization_loss

        cache_root = image_cache_dir
        if cache_root is None:
            cache_root = Path(self.model_dir) / "smiles_png_cache_MQMOP"
        self.image_cache = MQMOPImageCache(cache_dir=cache_root,
                                           export_image_size=export_image_size,
                                           input_image_size=input_image_size)

    def _regularization_loss(self) -> torch.Tensor:
        l1_regularization = torch.tensor(0., requires_grad=True)
        l2_regularization = torch.tensor(0., requires_grad=True)
        for name, param in self.model.named_parameters():
            if "bias" not in name:
                l1_regularization = l1_regularization + torch.norm(param, p=1)
                l2_regularization = l2_regularization + torch.norm(param, p=2)
        l1_norm = self.weight_decay * l1_regularization
        l2_norm = self.weight_decay * l2_regularization
        return l1_norm + l2_norm

    def default_generator(self,
                          dataset,
                          epochs: int = 1,
                          mode: str = "fit",
                          deterministic: bool = True,
                          pad_batches: bool = True):
        for _ in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                yield ([X_b, ids_b], [y_b], [w_b])

    def _smiles_batch_to_images(
            self, image_id_batch: Sequence[Any]) -> torch.Tensor:
        image_tensors: List[torch.Tensor] = []
        for image_id in image_id_batch:
            csv_row_number, smiles = parse_csv_row_smiles_image_id_mqmop(
                image_id)
            image_tensors.append(
                self.image_cache.load_tensor(smiles,
                                             csv_row_number=csv_row_number))
        return torch.stack(image_tensors, dim=0).to(self.device)

    def _prepare_batch(
        self, batch: Tuple[List[Any], List[torch.Tensor], List[torch.Tensor]]
    ) -> Tuple[List[Any], List[torch.Tensor], List[torch.Tensor]]:
        inputs, labels, weights = batch
        graph_inputs = inputs[0]
        image_id_inputs = inputs[1]
        dgl_graphs = [
            graph.to_dgl_graph(self_loop=self._self_loop)
            for graph in graph_inputs
        ]
        g = dgl.batch(dgl_graphs).to(self.device)
        images = self._smiles_batch_to_images(image_id_inputs)
        _, labels, weights = super(MQMOPModel, self)._prepare_batch(
            ([], labels, weights))
        return [g, images], labels, weights


__all__ = [
    "CustomMQMOPGNN",
    "MQMOP",
    "MQMOPModel",
]
