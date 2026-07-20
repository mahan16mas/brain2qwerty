from xp_config import experiment_config
from neuraltrain.models import BaseModelConfig as ModelConfig
from torch import nn
import torch
import numpy as np
from utils.augmentation import GaussianSmoothing
import torch 
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GEGLU(nn.Module):
    """Gated Gaussian Error Linear Unit (GEGLU) activation function, as introduced in
    the paper "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).
    """

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    """A feed-forward network with GEGLU activation.

    Args:
        dim (int): Input and output dimension
        mult (int, optional): Multiplier for hidden dimension. Defaults to 4
        dropout (float, optional): Dropout probability. Defaults to 0.2
    """

    def __init__(self, dim, mult=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)

def generate_sinusoidal_position_embs(num_timesteps, dim):
    position = torch.arange(num_timesteps).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    pe = torch.empty(num_timesteps, dim)
    pe[:, 0:dim // 2] = torch.sin(position * div_term)
    pe[:, dim//2:] = torch.cos(position * div_term)
    return pe


def generate_unit_embs(num_neurons, dim):
    return torch.randn((num_neurons, dim))

class TransformerPatchEncoder(nn.Module):
    def __init__(
        self,
        num_neurons, chunk_size, # data properties
        dim_hidden, n_layers, n_heads,    # transformer properties
    ):
        """Initialize the neural net components"""
        super().__init__()

        self.num_neurons = num_neurons
        self.chunk_size = chunk_size 
        self.dim_hidden = dim_hidden

        # Create the position embeddings
        # Note that these are kept constant in this implementation, i.e. _not_ learnable
        # self.position_embeddings = nn.Parameter(
        #     data=generate_sinusoidal_position_embs(self.num_neurons, self.chunk_size),
        #     requires_grad=False, # can set to True here, but might overfit
        # )

        self.unit_embeddings = nn.Parameter(
            data=generate_unit_embs(self.num_neurons, dim=self.dim_hidden),
            requires_grad=True,
        )

        # Create the transformer layers:
        # each composed of the Attention and the feedforward (FFN) blocks
        self.transformer_layers = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=self.dim_hidden,
                    num_heads=n_heads,
                    batch_first=True,
                ),
                FeedForward(dim=self.dim_hidden),
            ])
            for _ in range(n_layers)
        ])

        # self.time_agg_out = nn.LazyLinear(1)
        self.time_agg_out = nn.Linear(self.dim_hidden, 1)

        self.read_in = nn.Linear(self.chunk_size, dim_hidden)
        
    def forward(self, x, chunk_id, session_id):
        """
        Shape of x: (K, N, C)
            K: number of chunks 
            N: number of neurons 
            C: chunk size

        chunk_id: [C] -> id of chunk w.r.t whole trial 
        session_id: [C] -> session id of the trial that this chunk refers to 
        """

        ## # Read-in: converts our input marix to transformer tokens; one token for each timestep
        ## x = self.readin(x)  # (B, T, N) -> (B, T, D)
        x = self.read_in(x)

        ########## x = x.permute(2, 1, 0) # (T, N, K) -> (K, N, T)

        # print('---- in patch encoder ----')
        # print(x.shape )

        # Add position embeddings to the tokens
        # x = x + self.position_embeddings[None, ...]  # -> (B, T, D)
        x = x + self.unit_embeddings[None, ...]
        # print(x.shape)

        # Transformer
        for attn, ffn in self.transformer_layers:
            x = x + attn(x, x, x, need_weights=False)[0]
            x = x + ffn(x)
        # print(x.shape)

        # x is (K, N, C)
        
        # print('before', x.shape)
        x = self.time_agg_out(x) # this reduces feature dim, not sequences 
        if x.ndim == 3:
            x = x.squeeze(2)  # Remove singleton dimension
        # print(x.shape) # should be [K, N]
        
        return x

def get_transformer(dim):
    cfg = experiment_config()
    transformer_config = ModelConfig(**cfg["transformer_config"])

    transformer_model = transformer_config.build(dim=dim)
    return transformer_model

class HamedMetaModel(nn.Module):
    def __init__(self, num_neurons, chunk_size, dim_hidden, num_classes, n_layers=2, n_heads=4):
        super().__init__()

        # self.model, self.transformer, hidden = get_models(num_neurons, conv_dropout=0.5, dropout_input=dropout_input)
        self.patch_encoder = TransformerPatchEncoder(num_neurons, chunk_size=chunk_size, dim_hidden=dim_hidden, n_layers=n_layers, n_heads=n_heads)

        self.transformer = get_transformer(num_neurons)

        self.linear = nn.Linear(num_neurons, num_classes)

    def _patch_forward(self, neuro, subject_id, channel_positions) -> torch.Tensor:
        return self.patch_encoder(neuro, None, None)
    
    def _transformer_forward(self, uids, y_pred: torch.Tensor) -> torch.Tensor:
        # print('in transformers forward')
        # print('y_pred', y_pred.shape)

        uids = uids.detach().cpu().numpy()
        unique_uids, first_idx = np.unique(uids, return_index=True)
        unique_uids = unique_uids[np.argsort(first_idx)]

        grouped = [
            torch.stack([y_pred[i] for i, s in enumerate(uids) if s == uid])
            for uid in unique_uids
        ]
        max_len = max(len(g) for g in grouped)
        x = torch.zeros(len(grouped), max_len, y_pred.shape[1], device=y_pred.device)
        mask = torch.zeros(len(grouped), max_len, device=y_pred.device)
        out_lengths = torch.zeros(len(grouped), device=y_pred.device)
        for i, g in enumerate(grouped):
            x[i, : len(g)] = g
            mask[i, : len(g)] = 1
            out_lengths[i] = len(g)

        # print('x.shape', x.shape)
        out = self.transformer(x, mask=mask.bool())
        # print('out.shape', out.shape)
        # print('---')
        return self.linear(out), out_lengths.long()

    def forward(self, neuro, subject_id, channel_positions, uids):
        # print('Neuro', neuro.shape)
        # neuro = self.smoother.forward(neuro)
        y_pred = self._patch_forward(neuro, subject_id, channel_positions)
        return self._transformer_forward(uids, y_pred)



if __name__=="__main__":
    # mmm = TransformerPatchEncoder(192, 4, 0, 2, 2)
    # model = HamedMetaModel(192, 32, 128)
    # print(model)
    # x = torch.randn([616, 192, 4])
    # sid = torch.randn([616])
    # cpos = torch.randn([616, 192, 2])
    # uids = torch.randn([616]) 
    # # torch.Size([616, 192, 4]) torch.Size([616]) torch.Size([616, 192, 2]) torch.Size([616])

    # y, l = model(x, sid, cpos, uids)
    # print('y, l', y.shape, l.shape)

    # temp = TransformerPatchEncoder(192, 4, 0, 1, 1)
    # # print(temp)
    # x = torch.randn([616, 192, 4])
    # y = temp(x, None, None)
    # print(y.shape)

    # model = HamedMetaModel(192, 4, 256, 32)
    # print(model)
    # x = torch.randn([616, 192, 4])
    # sid = torch.zeros([616])
    # cpos = torch.randn([616, 192, 2])
    # uids = torch.randn([616]) 
    # # torch.Size([616, 192, 4]) torch.Size([616]) torch.Size([616, 192, 2]) torch.Size([616])

    # y, l = model(x, sid, cpos, uids)
    # print('y, l', y.shape, l.shape)

    K = 64
    N = 192
    C = 4 
    model = HamedMetaModel(N, C, 256, 32, 1, 1)
    x = torch.randn([K, N, C])
    sid = torch.zeros([K])
    cpos = torch.randn([K, N, C])
    uids = torch.concat((torch.zeros([K//2]), torch.ones([K//2])))

    from torchinfo import summary
    print(summary(model, input_data=(x, sid, cpos, uids), 
        # col_names=(
        #     "input_size",
        #     "output_size",
        #     "num_params",
        #     "trainable",
        # ),
        # depth=10,
        verbose=1,)
        )
    