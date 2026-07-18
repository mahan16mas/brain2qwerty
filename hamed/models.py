from xp_config import experiment_config
from neuraltrain.models import BaseModelConfig as ModelConfig
from torch import nn
import torch
import numpy as np
from utils.augmentation import GaussianSmoothing

def get_models(n_in_channels, conv_dropout=0.5, dropout_input=0.2):
    cfg = experiment_config()
    cfg["brain_model_config"]["conv_dropout"] = conv_dropout
    cfg["brain_model_config"]["dropout_input"] = dropout_input


    brain_config = ModelConfig(**cfg["brain_model_config"])
    transformer_config = ModelConfig(**cfg["transformer_config"])

    hidden_dim = brain_config.hidden

    brain_model = brain_config.build(n_in_channels=n_in_channels, n_outputs=hidden_dim)
    transformer_model = transformer_config.build(dim=hidden_dim)
    return brain_model,transformer_model, hidden_dim

class MetaModel(nn.Module):
    def __init__(self, num_neurons, num_classes, hidden=2048, conv_dropout=0.5, dropout_input=0.2):
        super().__init__()

        self.model, self.transformer, hidden = get_models(num_neurons, conv_dropout=conv_dropout, dropout_input=dropout_input)
        self.linear = nn.Linear(hidden, num_classes)

    def _cnn_forward(self, neuro, subject_id, channel_positions) -> torch.Tensor:
        return self.model(neuro, None, None)

    def _transformer_forward(self, uids, y_pred: torch.Tensor) -> torch.Tensor:
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

        out = self.transformer(x, mask=mask.bool())
        return self.linear(out), out_lengths.long()

    def forward(self, neuro, subject_id, channel_positions, uids):
        # neuro = self.smoother.forward(neuro)
        y_pred = self._cnn_forward(neuro, subject_id, channel_positions)
        return self._transformer_forward(uids, y_pred)
