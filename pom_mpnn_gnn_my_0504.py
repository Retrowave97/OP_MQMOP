import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import NNConv
from dgllife.model.gnn import MPNNGNN


class CustomMPNNGNN_my(MPNNGNN):
    """
    Customized MPNNGNN layer based MPNNGNN layer in dgllife library.

    Additional options:
    -> toggle for residual in gnn layer
    -> choice for message aggregator type

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    This class performs message passing in MPNN
    and returns the updated node representations.
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum'):
        """
        Parameters
        ----------
        node_in_feats: int
            Size for the input node features.
        node_out_feats: int
            Size for the output node representations. Default to 64.
        edge_in_feats: int
            Size for the input edge features. Default to 128.
        edge_hidden_feats: int
            Size for the hidden edge representations.
        num_step_message_passing: int
            Number of message passing steps. Default to 6.
        residual: bool
            If true, adds residual layer to gnn layer
        message_aggregator_type: str
            message aggregator type, 'sum', 'mean' or 'max'
        """
        super(CustomMPNNGNN_my,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing)

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats), nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats))
        self.gnn_layer = NNConv(in_feats=node_out_feats,
                                out_feats=node_out_feats,
                                edge_func=edge_network,
                                aggregator_type=message_aggregator_type,
                                residual=residual)


class CustomMPNNGNN_my_v1(nn.Module):
    """
    A stronger MPNN variant that keeps the original interface but adds a few
    changes that often help molecular property prediction:

    * untied message-passing parameters across steps
    * a deeper edge network for NNConv
    * per-step LayerNorm
    * dropout after message passing
    * learnable weighted fusion over all message-passing steps
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 dropout: float = 0.10,
                 layer_norm: bool = True,
                 jk: str = 'attention'):
        super(CustomMPNNGNN_my_v1, self).__init__()

        if jk not in ['last', 'mean', 'max', 'attention']:
            raise ValueError("jk must be one of ['last', 'mean', 'max', 'attention']")

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
                self.norms.append(nn.LayerNorm(node_out_feats))

        if jk == 'attention':
            self.step_logits = nn.Parameter(
                torch.zeros(num_step_message_passing, dtype=torch.float32))

    def reset_parameters(self):
        """Reinitialize model parameters."""
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
        if hasattr(self, 'step_logits'):
            nn.init.zeros_(self.step_logits)

    def _combine_step_outputs(self, step_outputs):
        """Fuse representations from all message-passing steps."""
        if len(step_outputs) == 0:
            raise ValueError("step_outputs should contain at least one tensor")

        if self.jk == 'last':
            return step_outputs[-1]

        stacked_outputs = torch.stack(step_outputs, dim=0)
        if self.jk == 'mean':
            return stacked_outputs.mean(dim=0)
        if self.jk == 'max':
            return stacked_outputs.max(dim=0).values

        step_weights = torch.softmax(self.step_logits[:len(step_outputs)], dim=0)
        view_shape = (len(step_outputs),) + (1,) * (stacked_outputs.dim() - 1)
        return torch.sum(stacked_outputs * step_weights.view(view_shape), dim=0)

    def forward(self, g, node_feats, edge_feats):
        """Perform message passing and return updated node representations."""
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats
        step_outputs = []

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


class CustomMPNNGNN_my_v2(nn.Module):
    """
    Message-passing backbone for the multimodal MPNNPOM v2 model.

    This intentionally preserves the strengthened v1 graph branch design so the
    new model can add an image modality without regressing the original graph
    encoder behavior.
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 dropout: float = 0.10,
                 layer_norm: bool = True,
                 jk: str = 'attention'):
        super(CustomMPNNGNN_my_v2, self).__init__()

        if jk not in ['last', 'mean', 'max', 'attention']:
            raise ValueError("jk must be one of ['last', 'mean', 'max', 'attention']")

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
                self.norms.append(nn.LayerNorm(node_out_feats))

        if jk == 'attention':
            self.step_logits = nn.Parameter(
                torch.zeros(num_step_message_passing, dtype=torch.float32))

    def reset_parameters(self):
        """Reinitialize model parameters."""
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
        if hasattr(self, 'step_logits'):
            nn.init.zeros_(self.step_logits)

    def _combine_step_outputs(self, step_outputs):
        """Fuse representations from all message-passing steps."""
        if len(step_outputs) == 0:
            raise ValueError("step_outputs should contain at least one tensor")

        if self.jk == 'last':
            return step_outputs[-1]

        stacked_outputs = torch.stack(step_outputs, dim=0)
        if self.jk == 'mean':
            return stacked_outputs.mean(dim=0)
        if self.jk == 'max':
            return stacked_outputs.max(dim=0).values

        step_weights = torch.softmax(self.step_logits[:len(step_outputs)], dim=0)
        view_shape = (len(step_outputs),) + (1,) * (stacked_outputs.dim() - 1)
        return torch.sum(stacked_outputs * step_weights.view(view_shape), dim=0)

    def forward(self, g, node_feats, edge_feats):
        """Perform message passing and return updated node representations."""
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats
        step_outputs = []

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


class CustomMPNNGNN_my_v3(CustomMPNNGNN_my_v2):
    """
    Graph backbone for the residual-fusion multimodal model.

    v3 keeps the strengthened graph encoder from v2 unchanged so improvements
    can come specifically from a safer multimodal fusion strategy.
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 dropout: float = 0.10,
                 layer_norm: bool = True,
                 jk: str = 'attention'):
        super(CustomMPNNGNN_my_v3,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing,
                             residual=residual,
                             message_aggregator_type=message_aggregator_type,
                             dropout=dropout,
                             layer_norm=layer_norm,
                             jk=jk)


class CustomMPNNGNN_my_v4(CustomMPNNGNN_my_v1):
    """
    Graph backbone for v4.

    v4 intentionally stays as close as possible to the graph branch used in
    MPNNPOMModel_my_v1, since that single-branch baseline has already been
    validated to generalize well.
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 dropout: float = 0.10,
                 layer_norm: bool = True,
                 jk: str = 'attention'):
        super(CustomMPNNGNN_my_v4,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing,
                             residual=residual,
                             message_aggregator_type=message_aggregator_type,
                             dropout=dropout,
                             layer_norm=layer_norm,
                             jk=jk)


class CustomMPNNGNN_my_v4_C2(CustomMPNNGNN_my):
    """
    Baseline graph backbone wrapper for the C2-only ablation.

    This keeps the original MPNNPOM graph encoder behavior unchanged while the
    multimodal image branch and residual fusion are added elsewhere.
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum'):
        super(CustomMPNNGNN_my_v4_C2,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing,
                             residual=residual,
                             message_aggregator_type=message_aggregator_type)


class CustomMPNNGNN_my_v5(CustomMPNNGNN_my_v4):
    """
    Graph backbone for v5.

    v5 keeps the v4/v1-style graph encoder intact so the only new experimental
    change can come from the lightweight multi-scale image branch.
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 dropout: float = 0.10,
                 layer_norm: bool = True,
                 jk: str = 'attention'):
        super(CustomMPNNGNN_my_v5,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing,
                             residual=residual,
                             message_aggregator_type=message_aggregator_type,
                             dropout=dropout,
                             layer_norm=layer_norm,
                             jk=jk)


class CustomMPNNGNN_my_v6(CustomMPNNGNN_my_v4):
    """
    Graph backbone for v6.

    v6 keeps the v4/v1-style graph encoder unchanged so the only new
    experimental variable is the multi-view image branch.
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 dropout: float = 0.10,
                 layer_norm: bool = True,
                 jk: str = 'attention'):
        super(CustomMPNNGNN_my_v6,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing,
                             residual=residual,
                             message_aggregator_type=message_aggregator_type,
                             dropout=dropout,
                             layer_norm=layer_norm,
                             jk=jk)


class CustomMPNNGNN_my_v7(CustomMPNNGNN_my_v4):
    """
    Graph backbone for v7.

    v7 keeps the validated v4/v1-style graph encoder intact so the main
    experimental change remains concentrated in the image branch.
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 dropout: float = 0.10,
                 layer_norm: bool = True,
                 jk: str = 'attention'):
        super(CustomMPNNGNN_my_v7,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing,
                             residual=residual,
                             message_aggregator_type=message_aggregator_type,
                             dropout=dropout,
                             layer_norm=layer_norm,
                             jk=jk)


class CustomMPNNGNN_my_v8(CustomMPNNGNN_my_v4):
    """
    Image-conditioned graph backbone for v8.

    v8 keeps the validated v1/v4 graph encoder as the base, but injects image
    information into two high-leverage graph modules:

    * per-step LayerNorm outputs are modulated by image-conditioned FiLM
    * JK-attention step fusion is turned into image-conditioned step weighting

    This lets the image branch influence graph representation learning itself,
    so removing the image path is expected to hurt the graph branch more than in
    v4 where image information only enters after graph encoding.
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 dropout: float = 0.10,
                 layer_norm: bool = True,
                 jk: str = 'attention',
                 image_context_feats: int = 128,
                 image_condition_hidden_feats: int = 64,
                 image_conditioned_norm: bool = True,
                 image_conditioned_jk: bool = True):
        super(CustomMPNNGNN_my_v8,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing,
                             residual=residual,
                             message_aggregator_type=message_aggregator_type,
                             dropout=dropout,
                             layer_norm=layer_norm,
                             jk=jk)

        self.image_context_feats = image_context_feats
        self.image_condition_hidden_feats = max(
            1, int(image_condition_hidden_feats))
        self.image_conditioned_norm = image_conditioned_norm
        self.image_conditioned_jk = image_conditioned_jk

        if self.image_conditioned_norm:
            self.image_norm_adapters = nn.ModuleList()
            for _ in range(num_step_message_passing):
                adapter = nn.Sequential(
                    nn.Linear(image_context_feats,
                              self.image_condition_hidden_feats),
                    nn.ReLU(),
                    nn.Linear(self.image_condition_hidden_feats,
                              2 * node_out_feats),
                )
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
                self.image_norm_adapters.append(adapter)

        if self.image_conditioned_jk:
            self.image_step_adapter = nn.Sequential(
                nn.Linear(image_context_feats,
                          self.image_condition_hidden_feats),
                nn.ReLU(),
                nn.Linear(self.image_condition_hidden_feats,
                          num_step_message_passing),
            )
            nn.init.zeros_(self.image_step_adapter[-1].weight)
            nn.init.zeros_(self.image_step_adapter[-1].bias)

    def reset_parameters(self):
        super(CustomMPNNGNN_my_v8, self).reset_parameters()

        if self.image_conditioned_norm:
            for adapter in self.image_norm_adapters:
                for layer in adapter:
                    if isinstance(layer, nn.Linear):
                        layer.reset_parameters()
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)

        if self.image_conditioned_jk:
            for layer in self.image_step_adapter:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
            nn.init.zeros_(self.image_step_adapter[-1].weight)
            nn.init.zeros_(self.image_step_adapter[-1].bias)

    def _apply_image_conditioned_norm(self, g, feats, step_idx,
                                      image_context):
        if (not self.image_conditioned_norm or image_context is None or
                self.norms is None):
            return feats

        gamma_beta = self.image_norm_adapters[step_idx](image_context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = dgl.broadcast_nodes(g, gamma)
        beta = dgl.broadcast_nodes(g, beta)
        return feats * (1 + gamma) + beta

    def _combine_step_outputs_with_image(self, g, step_outputs,
                                         image_context):
        if len(step_outputs) == 0:
            raise ValueError("step_outputs should contain at least one tensor")

        if self.jk != 'attention' or not self.image_conditioned_jk or image_context is None:
            return self._combine_step_outputs(step_outputs)

        num_steps = len(step_outputs)
        base_logits = self.step_logits[:num_steps].unsqueeze(0)
        image_delta_logits = self.image_step_adapter(image_context)[:, :num_steps]
        step_weights = torch.softmax(base_logits + image_delta_logits, dim=1)

        fused_output = torch.zeros_like(step_outputs[0])
        for step_idx, step_output in enumerate(step_outputs):
            node_weight = dgl.broadcast_nodes(g,
                                              step_weights[:, step_idx:step_idx + 1])
            fused_output = fused_output + step_output * node_weight

        return fused_output

    def forward(self, g, node_feats, edge_feats, image_context=None):
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats
        step_outputs = []

        for step_idx, (gnn_layer, gru) in enumerate(
                zip(self.gnn_layers, self.update_grus)):
            messages = F.relu(gnn_layer(g, node_feats, edge_feats))
            messages = self.dropout(messages)

            updated_feats = gru(messages, hidden_feats)
            if self.norms is not None:
                updated_feats = self.norms[step_idx](updated_feats)
            updated_feats = self._apply_image_conditioned_norm(
                g, updated_feats, step_idx, image_context)
            updated_feats = self.dropout(updated_feats)

            node_feats = updated_feats
            hidden_feats = updated_feats
            step_outputs.append(updated_feats)

        return self._combine_step_outputs_with_image(g, step_outputs,
                                                     image_context)
