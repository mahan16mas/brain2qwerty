from .config.xp_config import debug_config, experiment_config

from neuraltrain.models import BaseModelConfig as ModelConfig
from torch import nn
import torch
import numpy as np

def get_models(n_in_channels, conv_dropout=0.5, dropout_input=0.2,mahan_model_params=False, time_agg_out = "att", cnn_hidden=2048, cnn_depth=8, cnn_initial_linear=512, transformer_head: int = 4, transformer_depth: int = 2):
    cfg = experiment_config(meta_default=not mahan_model_params)
    cfg["brain_model_config"]["conv_dropout"] = conv_dropout
    cfg["brain_model_config"]["dropout_input"] = dropout_input
    cfg["brain_model_config"]["time_agg_out"] = time_agg_out
    if not mahan_model_params: 
        cfg["brain_model_config"]["hidden"] = cnn_hidden
        cfg["brain_model_config"]["depth"] = cnn_depth
        cfg["brain_model_config"]["initial_linear"] = cnn_initial_linear
        cfg["transformer_config"]["depth"] = transformer_depth
        cfg["transformer_config"]["heads"] = transformer_head

    brain_config = ModelConfig(**cfg["brain_model_config"])
    transformer_config = ModelConfig(**cfg["transformer_config"])

    hidden_dim = brain_config.hidden

    brain_model = brain_config.build(n_in_channels=n_in_channels, n_outputs=hidden_dim)
    transformer_model = transformer_config.build(dim=hidden_dim)
    return brain_model,transformer_model, hidden_dim

class MetaModel(nn.Module):
    def __init__(self, num_neurons, num_classes, cnn_hidden=2048, cnn_depth=8, cnn_initial_linear=512, conv_dropout=0.5, dropout_input=0.2, mahan_model_params = False, time_agg_out: str = "att",  transformer_depth: int = 4, transformer_head: int = 2,        
                 # do_smoothing = False, smooth_width=2.0
                 ):
        super().__init__()

        self.model, self.transformer, hidden = get_models(num_neurons, conv_dropout=conv_dropout, dropout_input=dropout_input, mahan_model_params=mahan_model_params, time_agg_out = time_agg_out, cnn_hidden=cnn_hidden, cnn_depth=cnn_depth, cnn_initial_linear=cnn_initial_linear, transformer_head = transformer_head, transformer_depth = transformer_depth)
        self.linear = nn.Linear(hidden, num_classes)

        #### self.smoother = (GaussianSmoothing(num_neurons, 20, smooth_width, dim=1)) if do_smoothing else (nn.Identity())

    def _cnn_forward(self, neuro, subject_id, channel_positions) -> torch.Tensor:
        return self.model(neuro, None, None)

    def _transformer_forward(self, uids, y_pred: torch.Tensor) -> torch.Tensor:
        # y_pred: [K, D] where K is number of chunks 

        uids = uids.detach().cpu().numpy() 
        unique_uids, first_idx = np.unique(uids, return_index=True)
        # print(unique_uids, first_idx)
        unique_uids = unique_uids[np.argsort(first_idx)]
        # print('-----')
        # print('unique_uids', unique_uids)
        grouped = [
            torch.stack([y_pred[i] for i, s in enumerate(uids) if s == uid])
            for uid in unique_uids
        ]
        # print(len(grouped))
        # print(grouped[0].shape, grouped[1].shape)
                
        max_len = max(len(g) for g in grouped)
        x = torch.zeros(len(grouped), max_len, y_pred.shape[1], device=y_pred.device) # [B, T_max, D] where B is number of unique trials in the current batch of chunks
        mask = torch.zeros(len(grouped), max_len, device=y_pred.device) # [B, T_max]
        out_lengths = torch.zeros(len(grouped), device=y_pred.device) # [B]
        for i, g in enumerate(grouped):
            x[i, : len(g)] = g
            mask[i, : len(g)] = 1
            out_lengths[i] = len(g)

        out = self.transformer(x, mask=mask.bool())
        return self.linear(out), out_lengths.long()

    def forward(self, neuro, subject_id, channel_positions, uids):
        """
        neuro: (total_chunks_in_batch, n_channels, CHUNK_LENGTH)  flat, no padding
        neuro_len         : (B,)   chunk count per trial -- use to split `neuro` back into
                                    per-trial groups (sums to total_chunks_in_batch), e.g.:
                                        trial_chunks = torch.split(neuro, neuro_len.tolist(), dim=0)
        target            : (B, L_max)  padded with 0 (0 = CTC blank / padding, never a real class)
        target_len        : (B,)   true (pre-padding) L per item
        meta              : (B, 3) [subject_id, session, trial_id]  (raw `subject` field from events)
                                    -- still one row per TRIAL, unchanged
        subject_id        : (total_chunks_in_batch,)  from the real LabelEncoder extractor,
                                    expanded via repeat_interleave(neuro_len) to align with
                                    `neuro`'s flat chunk dimension -- one row per CHUNK now,
                                    matching the original pipeline's per-window repetition
        channel_positions : (total_chunks_in_batch, n_channels, 2)  same expansion as subject_id
                                    -- same sensor layout repeated per chunk, not per trial
        """
        # neuro = self.smoother.forward(neuro)
        y_pred = self._cnn_forward(neuro, subject_id, channel_positions)
        # print('cnn output', y_pred.shape)
        return self._transformer_forward(uids, y_pred)



from xp_config import experiment_config
from neuraltrain.models import BaseModelConfig as ModelConfig
import torch 

if __name__=="__main__":
    K = 4
    N = 2
    C = 4 

    X = torch.zeros(K, N, C)

    unit_embeds = torch.ones((N, C))

    print(X + unit_embeds[None, ...])
    exit()

    cfg = experiment_config()
    cfg["brain_model_config"]["conv_dropout"] = 0.5
    cfg["brain_model_config"]["dropout_input"] = 0.2    

    brain_config = ModelConfig(**cfg["brain_model_config"])
    hidden_dim = brain_config.hidden

    brain_model = brain_config.build(n_in_channels=192, n_outputs=hidden_dim)

    print(brain_model)

    x = torch.randn([616, 192, 4])
    brain_model(x, None, None)

    transformer_config = ModelConfig(**cfg["transformer_config"])
    transformer_model = transformer_config.build(dim=hidden_dim)

