from xp_config import experiment_config
from neuraltrain.models import BaseModelConfig as ModelConfig
from torch import nn
import torch
import numpy as np
from utils.augmentation import GaussianSmoothing
import torch.nn.functional as F

class RNN_decoder(nn.Module): 
    def __init__(self, input_size=2048, rnn_hidden=2048, rnn_layers=5, bidir=False, rnn_dr=0.4, ):
        super().__init__()
        current_dim = input_size
        # self.rnn = nn.GRU(
        #     input_size=current_dim, 
        #     hidden_size=rnn_hidden,
        #     num_layers=rnn_layers,
        #     batch_first=True, 
        #     bidirectional=bidir, 
        #     dropout=rnn_dr
        # )
        self.rnn = nn.LSTM(
            input_size=current_dim, 
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True, 
            bidirectional=bidir, 
            dropout=rnn_dr
        )
        current_dim = rnn_hidden * (2 if bidir else 1)

        self.rnn_output_dim = current_dim

    def forward(self, x, lengths):
        # x: [B, T, D]
        x, _ = self.rnn(x)
        return x, lengths

class ConvOnly(nn.Module):
    def __init__(self, num_neurons, num_classes, # hidden=2048,
                 conv_dropout=0.5, dropout_input=0.2, mahan_model_params = False):
        """rnn_hidden = 2048, rnn_layers=5, bidir=False, rnn_dr=0.4, 
        """
        super().__init__()

        cfg = experiment_config(meta_default=not mahan_model_params)
        cfg["brain_model_config"]["conv_dropout"] = conv_dropout
        cfg["brain_model_config"]["dropout_input"] = dropout_input
    
        brain_config = ModelConfig(**cfg["brain_model_config"])
        # transformer_config = ModelConfig(**cfg["transformer_config"])
    
        hidden_dim = brain_config.hidden # cnn_hidden output
    
        self.patch_encoder = brain_config.build(n_in_channels=num_neurons, n_outputs=hidden_dim)

        self.linear = nn.Linear(hidden_dim, num_classes)

    def _cnn_forward(self, neuro, subject_id, channel_positions) -> torch.Tensor:
        return self.patch_encoder(neuro, None, None)

    def _decoder_forward(self, uids, y_pred: torch.Tensor) -> torch.Tensor:
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

        out_lengths = out_lengths.long()
        # print('before linear', x.shape)
        outputs = self.linear(x)
        return outputs, out_lengths 

    def forward(self, neuro, subject_id, channel_positions, uids):
        # neuro = self.smoother.forward(neuro)
        y_pred = self.patch_encoder(neuro, subject_id, channel_positions)
        # print('cnn output', y_pred.shape)
        return self._decoder_forward(uids, y_pred)

class ConvRNN(nn.Module):
    def __init__(self, num_neurons, num_classes, cnn_hidden=2048,
                 conv_dropout=0.5, dropout_input=0.2, mahan_model_params = False,
                 rnn_hidden = 2048, rnn_layers=5, bidir=False, rnn_dr=0.4):
        """rnn_hidden = 2048, rnn_layers=5, bidir=False, rnn_dr=0.4, 
        """
        super().__init__()

        cfg = experiment_config(meta_default=not mahan_model_params)
        cfg["brain_model_config"]["conv_dropout"] = conv_dropout
        cfg["brain_model_config"]["dropout_input"] = dropout_input
        cfg["brain_model_config"]["hidden"] = cnn_hidden
    
        brain_config = ModelConfig(**cfg["brain_model_config"])
        # transformer_config = ModelConfig(**cfg["transformer_config"])
    
        hidden_dim = brain_config.hidden # cnn_hidden output
    
        self.patch_encoder = brain_config.build(n_in_channels=num_neurons, n_outputs=hidden_dim)

        # if mahan_model_params:
        #     RNN = {
        #         # making it big enough so the RNN decoder is kinda similar in terms of number of parameters to the default meta transformer 
        #         "rnn_hidden": 1024, 
        #         "rnn_layers": 3, 
        #         "bidir": False, 
        #         "rnn_dr": 0.4, 
        #     }
        # else: 
        #     RNN = {
        #         # making it big enough so the RNN decoder is kinda similar in terms of number of parameters to the default meta transformer 
        #         "rnn_hidden": 2048, 
        #         "rnn_layers": 5, 
        #         "bidir": False, 
        #         "rnn_dr": 0.4, 
        #     }
        RNN = {
            "rnn_hidden": rnn_hidden, "rnn_layers": rnn_layers, 'bidir': bidir, "rnn_dr": rnn_dr,
        }
        self.rnn_decoder = RNN_decoder(input_size=hidden_dim, rnn_hidden=RNN["rnn_hidden"], rnn_layers=RNN["rnn_layers"], bidir=RNN["bidir"], rnn_dr=RNN["rnn_dr"])
    
        self.linear = nn.Linear(self.rnn_decoder.rnn_output_dim, num_classes)

    def _cnn_forward(self, neuro, subject_id, channel_positions) -> torch.Tensor:
        return self.patch_encoder(neuro, None, None)

    def _decoder_forward(self, uids, y_pred: torch.Tensor) -> torch.Tensor:
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

        out_lengths = out_lengths.long()
        # out = self.transformer(x, mask=mask.bool())
        rnn_out, out_lengths = self.rnn_decoder(x, out_lengths)
        return self.linear(rnn_out), out_lengths 

    def forward(self, neuro, subject_id, channel_positions, uids):
        # neuro = self.smoother.forward(neuro)
        y_pred = self.patch_encoder(neuro, subject_id, channel_positions)
        # print('cnn output', y_pred.shape)
        return self._decoder_forward(uids, y_pred)

class CEBRACNN(nn.Module): 
    def __init__(self, num_neurons, initial_layer_size=512, cebra_num_units=256, cebra_num_outputs=64, cebra_model_name="Offset5Model", cebra_pad_mode="replicate"):
        super().__init__()
        self.cebra_pad_mode = cebra_pad_mode
        self.initial_linear = nn.Conv1d(num_neurons, initial_layer_size, 1)

        import sys
        sys.path.append('CEBRA-main')
        from cebra.models import Offset5Model, Offset36Dropoutv2

        """
        try this later: 
        @register("resample1-model", deprecated=True)
        @register("offset4-model-2x-subsample")
        class Resample1Model(_OffsetModel, ResampleModelMixin):
        """
        # self.cebra = Offset36Dropoutv2(initial_layer_size, cebra_num_units, cebra_num_outputs)
        
        if cebra_model_name == "Offset5Model": 
            self.cebra = Offset5Model(initial_layer_size, cebra_num_units, cebra_num_outputs)
            self.left_pad, self.right_pad = self.cebra.get_offset().left, self.cebra.get_offset().right

        elif cebra_model_name == "Offset36Dropoutv2": 
            self.cebra = Offset36Dropoutv2(initial_layer_size, cebra_num_units, cebra_num_outputs)
            # self.left_pad, self.right_pad = self.cebra.get_offset().left, self.cebra.get_offset().right
            self.left_pad, self.right_pad = 3, 2
            self.cebra_pad_mode = 'replicate'
        else:
            print('in progress, gonna encounter erorr after this line')

        time_agg_out = "linear" # a safe option for testing
        if time_agg_out == "gap":
            self.time_agg_out = nn.AdaptiveAvgPool1d(1)
        elif time_agg_out == "linear":
            self.time_agg_out = nn.LazyLinear(1)
        elif time_agg_out == "att":
            from neuraltrain.models.common import BahdanauAttention
            self.time_agg_out = BahdanauAttention(input_size=cebra_num_outputs, hidden_size=256)

        self.apply_pad_before_cebra = True  # based on model? 
        
    def _apply_cebra(self, x):

        if self.apply_pad_before_cebra: 
            x = F.pad(x, (self.left_pad, self.right_pad-1), mode=self.cebra_pad_mode) # reflect
        # x: [B, D, T_W_padded]
        # print('After self._apply_cebra F.pad(x, (18, 17), ', x.shape)

        # print('bef ceb', x.shape)
        x = self.cebra(x)  # (B, D, T)
        # TEMP FIX: 
        if len(x.shape) == 2: 
            x = x.unsqueeze(2)
        # print('after ceb', x.shape)

        return x
    
    def forward(self, x): 
        """
        x: [B, D, T]
        lengths: [K]

        B - number of samples (chunks)
        D - number of channels (neuron dimensions)
        T - time dimension (equals to chunk size 4)
        """
        x = self.initial_linear(x) # [K, initial_layer_size, C]
        # print(x.shape)
        x = self._apply_cebra(x)

        x = self.time_agg_out(x)

        if x.ndim == 3:
            x = x.squeeze(2)  # Remove singleton dimension

        return x 


class CEBRATransformer(nn.Module):
    def __init__(self, num_neurons, num_classes, initial_layer_size = 512, cebra_num_units = 256, cebra_num_outputs = 64, cebra_model_name="Offset5Model", cebra_pad_mode = "replicate", transformer_depth=4, transformer_head=2):
        super().__init__()

        self.patch_encoder = CEBRACNN(
            num_neurons = num_neurons,
            initial_layer_size = initial_layer_size,  
            cebra_num_units = cebra_num_units,
            cebra_num_outputs = cebra_num_outputs,
            cebra_model_name = cebra_model_name,
            cebra_pad_mode=cebra_pad_mode
        )

        cfg = experiment_config()
        cfg["transformer_config"]["depth"] = transformer_depth
        cfg["transformer_config"]["heads"] = transformer_head
        transformer_config = ModelConfig(**cfg["transformer_config"])

        self.transformer = transformer_config.build(dim=cebra_num_outputs)
        transformer_output_dim = cebra_num_outputs       
        self.linear = nn.Linear(transformer_output_dim, num_classes)

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
        # neuro = self.smoother.forward(neuro)
        y_pred = self.patch_encoder(neuro)
        # print('cnn output', y_pred.shape)
        return self._transformer_forward(uids, y_pred)


class CEBRARNN(nn.Module):
    def __init__(self, num_neurons, num_classes, initial_layer_size = 512, cebra_num_units = 256, cebra_num_outputs = 64, cebra_model_name="Offset5Model", cebra_pad_mode = "replicate", rnn_hidden = 2048, rnn_layers=5, bidir=False, rnn_dr=0.4):
        super().__init__()

        self.patch_encoder = CEBRACNN(
            num_neurons = num_neurons,
            initial_layer_size = initial_layer_size,  
            cebra_num_units = cebra_num_units,
            cebra_num_outputs = cebra_num_outputs,
            cebra_model_name = cebra_model_name,
            cebra_pad_mode=cebra_pad_mode
        )

        RNN = {
            "rnn_hidden": rnn_hidden, "rnn_layers": rnn_layers, 'bidir': bidir, "rnn_dr": rnn_dr,
        }
        self.rnn_decoder = RNN_decoder(input_size=cebra_num_outputs, rnn_hidden=RNN["rnn_hidden"], rnn_layers=RNN["rnn_layers"], bidir=RNN["bidir"], rnn_dr=RNN["rnn_dr"])
            
        self.linear = nn.Linear(self.rnn_decoder.rnn_output_dim, num_classes)

    def _decoder_forward(self, uids, y_pred: torch.Tensor) -> torch.Tensor:
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

        out_lengths = out_lengths.long()
        # out = self.transformer(x, mask=mask.bool())
        rnn_out, out_lengths = self.rnn_decoder(x, out_lengths)
        return self.linear(rnn_out), out_lengths 


    def forward(self, neuro, subject_id, channel_positions, uids):
        # neuro = self.smoother.forward(neuro)
        y_pred = self.patch_encoder(neuro)
        # print('cnn output', y_pred.shape)
        return self._decoder_forward(uids, y_pred)


if __name__=="__main__":
    from torchinfo import summary
    # import torch.nn as nn 
    # import torch 
    
    # K = 64 # num chunks
    # N = 192
    # C = 4 # chunk size
    # x = torch.randn([K, N, C])
    # sid = torch.zeros([K])
    # cpos = torch.randn([K, N, C])
    # uids = torch.concat((torch.zeros([K//2]), torch.ones([K//2])))
    # # lengths = torch.randint(0, T_max, (B, )) + 1 
    # model = ConvOnly(192, 32)
    # # print(model)
    # out, l = model(x, sid, cpos, uids)
    # print(out.shape)
    # exit()

    # import torch.nn as nn 
    # import torch 
    
    # K = 64 # num chunks
    # N = 192
    # C = 4 # chunk size
    # x = torch.randn([K, N, C])
    # sid = torch.zeros([K])
    # cpos = torch.randn([K, N, C])
    # uids = torch.concat((torch.zeros([K//2]), torch.ones([K//2])))
    # # lengths = torch.randint(0, T_max, (B, )) + 1 
    # model = CEBRACNN(192, )
    # # print(model)
    # out = model(x)
    # print(out.shape)
    # exit()

    # import torch.nn as nn 
    # import torch 
    
    # K = 64 # num chunks
    # N = 192
    # C = 4 # chunk size
    # x = torch.randn([K, N, C])
    # sid = torch.zeros([K])
    # cpos = torch.randn([K, N, C])
    # uids = torch.concat((torch.zeros([K//2]), torch.ones([K//2])))
    # # lengths = torch.randint(0, T_max, (B, )) + 1 
    # model = CEBRATransformer(192, 32, cebra_num_units=1024, cebra_num_outputs = 1024, transformer_depth=4, transformer_head=2)
    # # print(model)
    # out, l = model(x, sid, cpos, uids)
    # print(out.shape)

    # print(summary(model, input_data=(x, sid, cpos, uids), 
    # # col_names=(
    # #     "input_size",
    # #     "output_size",
    # #     "num_params",
    # #     "trainable",
    # # ),
    # # depth=10,
    # verbose=1,)
    # )
    # exit()

    
    import torch.nn as nn 
    import torch 
    
    K = 10 # num chunks
    N = 192
    C = 32 # chunk size
    x = torch.randn([K, N, C])
    sid = torch.zeros([K])
    cpos = torch.randn([K, N, C])
    uids = torch.concat((torch.zeros([K//2]), torch.ones([K//2])))
    # lengths = torch.randint(0, T_max, (B, )) + 1 
    ### model = CEBRARNN(192, 32, cebra_num_units=1024, cebra_num_outputs = 1024, rnn_hidden=128, rnn_layers=1, cebra_model_name="Offset36Dropoutv2")
    model = CEBRATransformer(192, 32, initial_layer_size=512, cebra_num_units=2048, cebra_num_outputs = 2048, transformer_depth = 4, transformer_head= 2, cebra_model_name="Offset36Dropoutv2")
    """
    python start_trainer.py --cebra_patch_encoder --out_dir nlp21_meta_CEBRATRANSFORMER_(Offset45_h-2048_o-2048_tr-d-4_tr-h-2)_bs16_chunk32 --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name Offset36Dropoutv2 --cebra_hidden_dim 2048 --cebra_out_dim 2048 --cebra_pad_mode replicate --transformer_depth 4 --transformer_head 2 --chunk_size 32 --chunk_stride 4
    """
    # print(model)
    out, l = model(x, sid, cpos, uids)
    print(out.shape)

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
    exit()

    import torch.nn as nn 
    import torch 
    
    K = 64 # num chunks
    N = 192
    C = 4 # chunk size
    x = torch.randn([K, N, C])
    sid = torch.zeros([K])
    cpos = torch.randn([K, N, C])
    uids = torch.concat((torch.zeros([K//2]), torch.ones([K//2])))
    model = ConvRNN(N, 10, mahan_model_params=True)
    print(model.patch_encoder)

    # y_pred = model.model(x, sid, uids)
    # print(y_pred.shape)
    out, _ = model(x, sid, cpos, uids)
    print(out.shape)

    # out, _ = model(x, sid, cpos, uids)
    # print(out.shape, '\n')
        
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
