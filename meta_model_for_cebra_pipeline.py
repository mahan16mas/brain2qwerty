from xp_config import experiment_config
from neuraltrain.models import BaseModelConfig as ModelConfig
from torch import nn
import torch
import numpy as np
from utils.augmentation import GaussianSmoothing


class Unfolder(nn.Module):
    def __init__(self, kernel, stride):
        super().__init__()
        self.unfolder = torch.nn.Unfold(
            (kernel, 1), dilation=1, padding=0, stride=stride
        )
        self.kernel = kernel
        self.stride = stride
    
    def forward(self, x, lengths):
        x = torch.permute(
                self.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
                (0, 2, 1),
            )
        lengths = ((lengths - self.kernel) / self.stride).to(torch.int32)
        # lengths = torch.div(
        #     lengths - self.kernel,
        #     self.stride,
        #     rounding_mode="floor"
        # ) + 1
        return x, lengths

class AvgPool(nn.Module):
    def __init__(self, kernel, stride):
        super().__init__()

        self.pool = nn.AvgPool1d(
            kernel_size=kernel,
            stride=stride,
            padding=0,
            ceil_mode=False,
        )

        self.kernel = kernel
        self.stride = stride

    def forward(self, x, lengths):
        # x: [B, T_max, D]

        x = x.permute(0, 2, 1)       # [B, D, T_max]
        x = self.pool(x)              # [B, D, T_new_max]
        x = x.permute(0, 2, 1)       # [B, T_new_max, D]

        lengths = torch.div(
            lengths - self.kernel,
            self.stride,
            rounding_mode="floor",
        ) + 1

        return x, lengths

def get_models(n_in_channels, conv_dropout=0.5, dropout_input=0.2,mahan_model_params=False, time_agg_out = "att", cnn_hidden=2048, cnn_depth=8, cnn_initial_linear=512, cnn_output = 2048, transformer_head: int = 4, transformer_depth: int = 2, unfolding: str = 'CEBRA_32_4'):
    cfg = experiment_config(meta_default=not mahan_model_params)

    # since we dont need time aggregation in this pipeline
    cfg["brain_model_config"]["name"] = "SimpleConv"
    #### cfg["brain_model_config"]["backbone_out_channels"] = 128
    cfg["brain_model_config"].pop("time_agg_out") 
    if cnn_initial_linear == 0: 
        cfg["brain_model_config"].pop("initial_linear") 
        
    cfg["brain_model_config"]["conv_dropout"] = conv_dropout
    cfg["brain_model_config"]["dropout_input"] = dropout_input
    
    if not mahan_model_params: 
        cfg["brain_model_config"]["hidden"] = cnn_hidden
        cfg["brain_model_config"]["depth"] = cnn_depth
        cfg["brain_model_config"]["initial_linear"] = cnn_initial_linear
        cfg["transformer_config"]["depth"] = transformer_depth
        cfg["transformer_config"]["heads"] = transformer_head

    brain_config = ModelConfig(**cfg["brain_model_config"])
    transformer_config = ModelConfig(**cfg["transformer_config"])

    brain_model = brain_config.build(n_in_channels=n_in_channels, n_outputs=cnn_output)

    if unfolding == "CEBRA_32_4": 
        unfolder = Unfolder(32, 4)
        transformer_dim = 32 * cnn_output
    elif unfolding == "CEBRA_4_4": 
        unfolder = Unfolder(4, 4)
        transformer_dim = 4 * cnn_output
    elif unfolding == "AVGPOOL_4_4": 
        unfolder = AvgPool(4, 4)
        transformer_dim = cnn_output * 1 
    elif unfolding == "AVGPOOL_25_4": 
        unfolder = AvgPool(25, 4)
        transformer_dim = cnn_output * 1 
    
    transformer_model = transformer_config.build(dim=transformer_dim)
    output_dim = transformer_dim
    return brain_model,transformer_model, output_dim, unfolder

class MetaModel(nn.Module):
    def __init__(self, num_neurons, num_classes, cnn_hidden=2048, cnn_depth=8, cnn_initial_linear=512, cnn_output = 2048, conv_dropout=0.5, dropout_input=0.2, mahan_model_params = False, time_agg_out: str = "att",  transformer_depth: int = 4, transformer_head: int = 2, unfolding: str = "CEBRA_32_4",       
                 # do_smoothing = False, smooth_width=2.0
                 ):
        super().__init__()

        self.model, self.transformer, output_dim, self.unfolder = get_models(num_neurons, conv_dropout=conv_dropout, dropout_input=dropout_input, mahan_model_params=mahan_model_params, time_agg_out = time_agg_out, cnn_hidden=cnn_hidden, cnn_depth=cnn_depth, cnn_initial_linear=cnn_initial_linear, cnn_output=cnn_output, transformer_head = transformer_head, transformer_depth = transformer_depth, unfolding=unfolding)
        self.linear = nn.Linear(output_dim, num_classes)

        #### self.smoother = (GaussianSmoothing(num_neurons, 20, smooth_width, dim=1)) if do_smoothing else (nn.Identity())

    def _cnn_forward(self, neuro, lengths, subject_id, channel_positions) -> torch.Tensor:
        # no padding, or max pooling, so dont change the `lengths`
        neuro = neuro.permute(0, 2, 1) # [B, D, T_max] 
        return self.model(neuro, None, None), lengths

    def _unfolding(self, x, lengths): 
        # [B, T, D]
        x, lengths = self.unfolder(x, lengths) # [B, T//stride, D*kernel]
        return x, lengths
    
    def _transformer_forward(self, x: torch.Tensor, lengths) -> torch.Tensor:
        # x: [B, T, D] ? 
        
        # print(max(lengths), x.shape[1])
        # assert max(lengths) == x.shape[1], 'must be equal, if not two possibilities: 1) the data loading is broken (returned length from dataloader doesnt match the T_max dim) 2) updating the length throughout the model is broken'

        B = x.shape[0]
        mask = torch.zeros(B, x.shape[1], device=x.device) # [B, T_max]
        for i, l in enumerate(lengths):
            mask[i, : l] = 1

        out = self.transformer(x, mask=mask.bool())
        return self.linear(out), lengths

    def forward(self, neuro, lengths):
        ### Since this model is called with cebra script: 
        # neuro : [B, T_max, D]
        # lengths : [B]

        # neuro = self.smoother.forward(neuro) # since we dont add noise, and do the gaussian smoothing when loading, this is not necessary right now, TODO: unless you add noise augmentation 

        cnn_out, lengths = self._cnn_forward(neuro, lengths, None, None) # you might wanna pass lengths if you do padding later 
        cnn_out = cnn_out.permute(0, 2, 1) # [B, T, ceb_out]

        x, lengths = self._unfolding(cnn_out, lengths)

        out, lengths = self._transformer_forward(x, lengths)
        return out, lengths, None, None 


if __name__=="__main__":
    import torch.nn as nn 
    import torch 
    
    B = 4
    T_max = 1000
    D = 192
    x = torch.randn((B, T_max, D))
    lengths = torch.randint(0, T_max, (B, )) + 1 
    # model = MetaModel(N, 32, mahan_model_params=False, time_agg_out="gap", cnn_hidden=2048, cnn_depth=8, cnn_initial_linear=512, transformer_depth=4, transformer_head=2) # 'gap', 'linear', 'att'
    model = MetaModel(D, 32, mahan_model_params=False, time_agg_out="gap", cnn_hidden=256, cnn_depth=8, cnn_initial_linear=0, cnn_output=64, transformer_depth=4, transformer_head=2, unfolder_kernel= 32, unfolder_stride= 4) 
    print(model)
    
    y, l = model(x, lengths)
    print(y.shape, l)

    from torchinfo import summary 
    print(summary(model, input_data=(x,  lengths), verbose=1,))

    exit()

    from torchinfo import summary
    print(
        summary(
            model.model, input_data=(x, sid, cpos), 
            # col_names=(
            #     "input_size",
            #     "output_size",
            #     "num_params",
            #     "trainable",
            # ),
            # depth=10,
            depth= 4, # 3
            verbose=0,
            )
        )
    
    exit()
    # print(out.shape)

    # out, _ = model(x, sid, cpos, uids)
    # print(out.shape, '\n')
        
    exit()

    """
    ------------------------- Model Explanation ------------------------
    K - number of samples (chunks)
    N - number of channels (neuron dimensions)
    C - time dimension (equals to chunk size 4)

    (K, N, C) -> (K, 512(N2), C) First linear layer: Conv1d(out=512, k=1, s=1):
        A kernel size of 1 means the convolution examines one time step at a time. At each time step, it takes all 192 input-channel values and linearly transforms them into 512 values. essentially nn.Linear(192, 512)
    (K, N2, C) -> (K, N3, C) encoder: 
        first conv module: 
            an input dropout (0.2)
            Conv1d(512, 2048, kernel_size=(3,), stride=(1,), padding=(1,)): 
                It mixes information across all 512 input channels.
                It combines information from neighboring time steps using a window of size 3.
                For output position t, each output channel looks at approximately: t - 1, t, t + 1 (across all 512 input channels.)
                output[b, channel, t]
                    = weighted combination of
                    input[b, all 512 channels, t-1:t+2]
            followed by batchnorm1d, gelu, and another 0.5 dropout 
        and then 7x conv modules as follows: 
            Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(1,))
                At each time step t, every output channel is computed from: all 2048 input channels at times t-1, t, and t+1
                So this is effectively a temporal feature-processing layer: it preserves the tensor shape but transforms the features using a three-step temporal receptive field.
            followed by bn1d, gelu, dropout 0.5 and a LayerScale layer (This rescales diagonally residual outputs close to 0 initially, then learnt)
        Note that in this 7x conv modules, 
            first one has k=3,s=1,p=2,d=2 (with k=3 and dilation=2, 
                the three kernel positions are spaced two time steps apart. For output position t, the layer uses: 
                x[:, :, t - 2]
                x[:, :, t]
                x[:, :, t + 2]
                rather than the ordinary convolution positions:
                x[:, :, t - 1]
                x[:, :, t]
                x[:, :, t + 1]
            )
            second one: Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
                At output time step t, it uses three input positions:
                t - 4,  t,  t + 4
            third one: Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(1,))
            fourth: Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,)) 
                At output time t, it uses three input positions:
                t - 2,  t,  t + 2
            fifth: Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
            sixth: Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(1,))
            seventh: Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    (K, N3, C) -> (K, N3) time_agg_out (att): 
        (time_agg_out): BahdanauAttention(
            (Wa): LazyLinear(in_features=0, out_features=256, bias=True)
            (Ua): LazyLinear(in_features=0, out_features=256, bias=True)
            (Va): Linear(in_features=256, out_features=1, bias=True)
        )
        This layer performs attention pooling over the time dimension.
        The attention module learns how important each of the four time steps is, then produces a weighted average of them.
            Assume 
                input_size = 2048
                hidden_size = H
            1. Move the feature dimension last
                keys = keys.transpose(2, 1)
                Shape transformation:
                [64, 2048, 4] → [64, 4, 2048]
                Now each time step is represented by a 2048-dimensional vector:
                    keys[b, 0, :]  # features at time 0
                    keys[b, 1, :]  # features at time 1
                    keys[b, 2, :]  # features at time 2
                    keys[b, 3, :]  # features at time 3
            2. Project every time step into the attention hidden space
                PyTorch applies this linear layer independently to every time step:
                [64, 4, 2048] → [64, 4, H]
                The linear layer mixes the 2048 channels, but does not mix different time steps.
            3. Optionally incorporate the queries
                When queries=None, as may be the case in your network, the score is computed from the keys alone:
                sum_[b, t] = Wa(keys[b, t])
                In that situation, this acts more like learned temporal attention pooling than the standard encoder-decoder form of Bahdanau attention.
            4. Calculate one score per time step
                scores = self.Va(torch.tanh(sum_))
                    1. tanh: [64, 4, H] → [64, 4, H]
                    2. self.Va = nn.Linear(H, 1)
                        [64, 4, H] → [64, 4, 1]
                        Each of the four time steps now has one scalar importance score:
                            batch 0: [score₀, score₁, score₂, score₃]
                            batch 1: [score₀, score₁, score₂, score₃]
                            ...
                scores = scores.squeeze(2).unsqueeze(1)
                    changes the shape:
                    [64, 4, 1]
                        ↓ squeeze(2)
                    [64, 4]
                        ↓ unsqueeze(1)
                    [64, 1, 4]
            5. Normalize the scores into attention weights
                weights = nn.functional.softmax(scores, dim=-1)
                    [64, 1, 4]
                    For each input in the batch, the four weights sum to one:
                        w₀ + w₁ + w₂ + w₃ = 1
            6. Compute a weighted average over time
                At this point:
                    weights: [64, 1, 4]
                    keys:    [64, 4, 2048]
                The operation: context = torch.bmm(weights, keys) performs batched matrix multiplication:
                    [64, 1, 4] × [64, 4, 2048] → [64, 1, 2048]
                    For every feature channel f:
                        context[b, 0, f] = (
                            weights[b, 0, 0] * keys[b, 0, f]
                            + weights[b, 0, 1] * keys[b, 1, f]
                            + weights[b, 0, 2] * keys[b, 2, f]
                            + weights[b, 0, 3] * keys[b, 3, f]
                        )
                So all 2048 features use the same four temporal attention weights.
            7. Restore channels-first format
                context = context.transpose(2, 1)
                    [64, 1, 2048] → [64, 2048, 1]

        Overall interpretation

            The transformation is:
                Input:  4 vectors, each containing 2048 features
                                    ↓
                Learn one importance weight for each time step
                                    ↓
                Take a weighted average of the 4 vectors
                                    ↓
                Output: one vector containing 2048 features

            In compact form:
                [64, 2048, 4]
                        ↓ attention over the 4 time steps
                [64, 2048, 1]
                        ↓ squeeze
                [64, 2048]

        So the layer compresses the time dimension from four time steps into one summary vector, while retaining all 2048 feature channels. It learns whether each sample should emphasize, for example, the first, second, third, or fourth time step.

    ====================================================================================================
    Layer (type:depth-idx)                             Output Shape              Param #
    ====================================================================================================
    MetaModel                                          [2, 32, 10]               --
    ├─SimpleConvTimeAggModel: 1-1                      [64, 2048]                --
    │    └─Sequential: 2-1                             [64, 512, 4]              --
    │    │    └─Conv1d: 3-1                            [64, 512, 4]              98,816
    │    └─ConvSequence: 2-2                           [64, 2048, 4]             --
    │    │    └─ModuleList: 3-2                        --                        91,285,504
    │    └─BahdanauAttention: 2-3                      [64, 2048, 1]             524,544
    │    │    └─Linear: 3-3                            [64, 4, 256]              524,544
    │    │    └─Linear: 3-4                            [64, 4, 1]                257
    ├─Encoder: 1-2                                     [2, 32, 2048]             --
    │    └─RotaryEmbedding: 2-4                        [1, 32, 512]              --
    │    └─ModuleList: 2-5                             --                        --
    │    │    └─ModuleList: 3-5                        --                        16,779,265
    │    │    └─ModuleList: 3-6                        --                        33,566,721
    │    │    └─ModuleList: 3-7                        --                        16,779,265
    │    │    └─ModuleList: 3-8                        --                        33,566,721
    │    │    └─ModuleList: 3-9                        --                        16,779,265
    │    │    └─ModuleList: 3-10                       --                        33,566,721
    │    │    └─ModuleList: 3-11                       --                        16,779,265
    │    │    └─ModuleList: 3-12                       --                        33,566,721
    │    └─ScaleNorm: 2-6                              [2, 32, 2048]             1
    ├─Linear: 1-3                                      [2, 32, 10]               20,490
    ====================================================================================================
    Total params: 293,838,100
    Trainable params: 293,838,100
    Non-trainable params: 0
    Total mult-adds (Units.GIGABYTES): 23.82

MetaModel(
  (model): SimpleConvTimeAggModel(
    (initial_linear): Sequential(
      (0): Conv1d(192, 512, kernel_size=(1,), stride=(1,))
    )
    (encoder): ConvSequence(
      (sequence): ModuleList(
        (0): Sequential(
          (0): Dropout(p=0.2, inplace=False)
          (1): Conv1d(512, 2048, kernel_size=(3,), stride=(1,), padding=(1,))
          (2): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): GELU(approximate='none')
          (4): Dropout(p=0.5, inplace=False)
        )
        (1): Sequential(
          (0): Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
          (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.5, inplace=False)
          (4): LayerScale()
        )
        (2): Sequential(
          (0): Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
          (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.5, inplace=False)
          (4): LayerScale()
        )
        (3): Sequential(
          (0): Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(1,))
          (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.5, inplace=False)
          (4): LayerScale()
        )
        (4): Sequential(
          (0): Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
          (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.5, inplace=False)
          (4): LayerScale()
        )
        (5): Sequential(
          (0): Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
          (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.5, inplace=False)
          (4): LayerScale()
        )
        (6): Sequential(
          (0): Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(1,))
          (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.5, inplace=False)
          (4): LayerScale()
        )
        (7): Sequential(
          (0): Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
          (1): LayerScale()
        )
      )
      (glus): ModuleList(
        (0-7): 8 x None
      )
    )
    (time_agg_out): BahdanauAttention(
      (Wa): Linear(in_features=2048, out_features=256, bias=True)
      (Ua): LazyLinear(in_features=0, out_features=256, bias=True)
      (Va): Linear(in_features=256, out_features=1, bias=True)
    )
  )
  (transformer): Encoder(
    (layers): ModuleList(
      (0): ModuleList(
        (0): ModuleList(
          (0): ScaleNorm()
          (1-2): 2 x None
        )
        (1): Attention(
          (to_q): Linear(in_features=2048, out_features=2048, bias=False)
          (to_k): Linear(in_features=2048, out_features=2048, bias=False)
          (to_v): Linear(in_features=2048, out_features=2048, bias=False)
          (split_q_heads): Rearrange('b n (h d) -> b h n d', h=2)
          (split_k_heads): Rearrange('b n (h d) -> b h n d', d=1024)
          (split_v_heads): Rearrange('b n (h d) -> b h n d', d=1024)
          (merge_heads): Rearrange('b h n d -> b n (h d)')
          (attend): Attend(
            (attn_dropout): Dropout(p=0.1, inplace=False)
          )
          (to_out): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (2): Residual()
      )
      (1): ModuleList(
        (0): ModuleList(
          (0): ScaleNorm()
          (1-2): 2 x None
        )
        (1): FeedForward(
          (ff): Sequential(
            (0): Sequential(
              (0): Linear(in_features=2048, out_features=8192, bias=True)
              (1): GELU(approximate='none')
            )
            (1): Dropout(p=0.0, inplace=False)
            (2): Linear(in_features=8192, out_features=2048, bias=True)
          )
        )
        (2): Residual()
      )
      (2): ModuleList(
        (0): ModuleList(
          (0): ScaleNorm()
          (1-2): 2 x None
        )
        (1): Attention(
          (to_q): Linear(in_features=2048, out_features=2048, bias=False)
          (to_k): Linear(in_features=2048, out_features=2048, bias=False)
          (to_v): Linear(in_features=2048, out_features=2048, bias=False)
          (split_q_heads): Rearrange('b n (h d) -> b h n d', h=2)
          (split_k_heads): Rearrange('b n (h d) -> b h n d', d=1024)
          (split_v_heads): Rearrange('b n (h d) -> b h n d', d=1024)
          (merge_heads): Rearrange('b h n d -> b n (h d)')
          (attend): Attend(
            (attn_dropout): Dropout(p=0.1, inplace=False)
          )
          (to_out): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (2): Residual()
      )
      (3): ModuleList(
        (0): ModuleList(
          (0): ScaleNorm()
          (1-2): 2 x None
        )
        (1): FeedForward(
          (ff): Sequential(
            (0): Sequential(
              (0): Linear(in_features=2048, out_features=8192, bias=True)
              (1): GELU(approximate='none')
            )
            (1): Dropout(p=0.0, inplace=False)
            (2): Linear(in_features=8192, out_features=2048, bias=True)
          )
        )
        (2): Residual()
      )
      (4): ModuleList(
        (0): ModuleList(
          (0): ScaleNorm()
          (1-2): 2 x None
        )
        (1): Attention(
          (to_q): Linear(in_features=2048, out_features=2048, bias=False)
          (to_k): Linear(in_features=2048, out_features=2048, bias=False)
          (to_v): Linear(in_features=2048, out_features=2048, bias=False)
          (split_q_heads): Rearrange('b n (h d) -> b h n d', h=2)
          (split_k_heads): Rearrange('b n (h d) -> b h n d', d=1024)
          (split_v_heads): Rearrange('b n (h d) -> b h n d', d=1024)
          (merge_heads): Rearrange('b h n d -> b n (h d)')
          (attend): Attend(
            (attn_dropout): Dropout(p=0.1, inplace=False)
          )
          (to_out): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (2): Residual()
      )
      (5): ModuleList(
        (0): ModuleList(
          (0): ScaleNorm()
          (1-2): 2 x None
        )
        (1): FeedForward(
          (ff): Sequential(
            (0): Sequential(
              (0): Linear(in_features=2048, out_features=8192, bias=True)
              (1): GELU(approximate='none')
            )
            (1): Dropout(p=0.0, inplace=False)
            (2): Linear(in_features=8192, out_features=2048, bias=True)
          )
        )
        (2): Residual()
      )
      (6): ModuleList(
        (0): ModuleList(
          (0): ScaleNorm()
          (1-2): 2 x None
        )
        (1): Attention(
          (to_q): Linear(in_features=2048, out_features=2048, bias=False)
          (to_k): Linear(in_features=2048, out_features=2048, bias=False)
          (to_v): Linear(in_features=2048, out_features=2048, bias=False)
          (split_q_heads): Rearrange('b n (h d) -> b h n d', h=2)
          (split_k_heads): Rearrange('b n (h d) -> b h n d', d=1024)
          (split_v_heads): Rearrange('b n (h d) -> b h n d', d=1024)
          (merge_heads): Rearrange('b h n d -> b n (h d)')
          (attend): Attend(
            (attn_dropout): Dropout(p=0.1, inplace=False)
          )
          (to_out): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (2): Residual()
      )
      (7): ModuleList(
        (0): ModuleList(
          (0): ScaleNorm()
          (1-2): 2 x None
        )
        (1): FeedForward(
          (ff): Sequential(
            (0): Sequential(
              (0): Linear(in_features=2048, out_features=8192, bias=True)
              (1): GELU(approximate='none')
            )
            (1): Dropout(p=0.0, inplace=False)
            (2): Linear(in_features=8192, out_features=2048, bias=True)
          )
        )
        (2): Residual()
      )
    )
    (layer_integrators): ModuleList(
      (0-7): 8 x None
    )
    (rotary_pos_emb): RotaryEmbedding()
    (rel_pos): AlibiPositionalBias()
    (adaptive_mlp): Identity()
    (final_norm): ScaleNorm()
    (skip_combines): ModuleList(
      (0-7): 8 x None
    )
  )
  (linear): Linear(in_features=2048, out_features=10, bias=True)
)
    """

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

