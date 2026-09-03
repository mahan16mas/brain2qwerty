import os 
import sys 
import inspect 
# relative import hacks (sorry)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # for bash user
os.chdir(parentdir)  # for pycharm user

import torch.nn as nn
import torch

from pathlib import Path
_current_dir = Path(__file__).parent.resolve()

PROJECT_ROOT_DIR = _current_dir.parent.resolve()
CEBRA_DIR = (PROJECT_ROOT_DIR / "CEBRA-main").resolve()

import sys
sys.path.append(str(CEBRA_DIR))
from cebra.models import Offset36Dropoutv2, Offset10Model # , Offset36Dropoutv2BN, Offset10ModelBN
import torch.nn.functional as F

import math
import numbers
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = torch.permute(input, (0, 2, 1))
        input = self.conv(input, weight=self.weight, groups=self.groups, padding="same")
        input = torch.permute(input, (0, 2, 1))
        return input


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
        # lengths = ((lengths - self.kernel) / self.stride).to(torch.int32)
        lengths = torch.div(
            lengths - self.kernel,
            self.stride,
            rounding_mode="floor"
        ) + 1
        return x, lengths


class Encoder_Decoder(nn.Module):
    """
    Input:  (B, T, F)
    Output: logits for CTC of shape (T, B, C)
    """

    def __init__(self, neural_dim, cebra_out_dim, kernel, stride, num_classes, rnn_hidden, rnn_layers, rnn_dr = 0.4, rnn_bidir=True, cebra_unfolder=False, gru = False, smooth_width=2.0, gauss_in=True, no_rnn=False,
                 cebra_window_10=False, cebra_bn = False, contrastive_on_decoder=False,
                 ceb_hidden=256, initial_layer_size = 0):
        super().__init__()
        self.ceb_hidden = ceb_hidden
        self.initial_layer_size = initial_layer_size
        def init_cebra(in_features):
            import sys
            sys.path.append('CEBRA-main')
            from cebra.models import Offset36Dropoutv2, Offset10Model # , Offset36Dropoutv2BN, Offset10ModelBN, Offset36Dropoutv205 
            if cebra_window_10:
                self.left_of = 5
                ceb_model = Offset10ModelBN if cebra_bn else Offset10Model
            else:
                self.left_of = 18
                ceb_model = Offset36Dropoutv2
            # return ceb_model(in_features, 256, cebra_out_dim)
            return ceb_model(in_features, self.ceb_hidden, cebra_out_dim)
        
        current_dim = neural_dim
        self.cebra_unfolder = cebra_unfolder
        self.contrastive_on_decoder = contrastive_on_decoder
        print('self.contrastive_on_decoder', self.contrastive_on_decoder)
        print('self.cebra_unfolder', self.cebra_unfolder)
        self.smoother = (GaussianSmoothing(neural_dim, 20, smooth_width, dim=1)) if gauss_in else (nn.Identity())

        ### 
        self.add_initial_layer = False if self.initial_layer_size == 0 else True 
        print('self.add_initial_layer', self.add_initial_layer)
        if self.add_initial_layer: 
            self.initial_layer = nn.Conv1d(current_dim, self.initial_layer_size, (1, ), (1, ))
            current_dim = self.initial_layer_size

        if cebra_unfolder:
            self.cebra = init_cebra(current_dim)
            current_dim = cebra_out_dim

        self.unfolder = Unfolder(kernel, stride)
        current_dim *= kernel

        if not cebra_unfolder:
            self.cebra = init_cebra(current_dim)
            current_dim = cebra_out_dim

        if not no_rnn:
            if gru:
                self.rnn = nn.GRU(
                    current_dim, 
                    rnn_hidden,
                    rnn_layers,
                    batch_first=True, 
                    bidirectional=rnn_bidir, 
                    dropout=rnn_dr
                    )
            else:
                self.rnn = nn.LSTM(
                    current_dim,
                    rnn_hidden, 
                    rnn_layers,
                    batch_first=True,
                    bidirectional=rnn_bidir, 
                    dropout=rnn_dr
                )
            current_dim = rnn_hidden * (2 if rnn_bidir else 1)
        else:
            self.rnn = lambda x: (x, None)
        
        self.final_decoder = nn.Linear(current_dim, num_classes)
    
    def _apply_cebra(self, x, lengths):
        """Helper to permute, pad, forward CEBRA, and permute back."""
        # print(x.shape)

        x = x.permute(0, 2, 1)  # (B, C, T)
        # print(x.shape)

        x = F.pad(x, (self.left_of, self.left_of - 1), mode='replicate')
        # print(x.shape)

        x = self.cebra(x).permute(0, 2, 1)  # (B, T, C)
        # print(x.shape)

        # self.embeddings = x
        # self.emb_lengths = lengths
        # print(self.embeddings.shape)
        return x
    
    def get_cebra_embs(self):
        return self.embeddings, self.emb_lengths
    
    def forward(self, x, lengths):
        # print(x.shape)
        x = self.smoother(x)
        # print('before init layer', x.shape)
        # print(x.shape)
        
        if self.add_initial_layer: 
            x = self.initial_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        # print('after inital layer', x.shape)

        # print('before cebra', x.shape)
        if self.cebra_unfolder:
            x = self._apply_cebra(x, lengths)
            embeddings = x
            emb_lengths = lengths
        # print('after cebra', x.shape)
        # print(x.shape)

        x, lengths = self.unfolder(x, lengths)
        # print('after unfolder', x.shape)
        # print(x.shape)

        if not self.cebra_unfolder:
            x = self._apply_cebra(x, lengths)
            embeddings = x
            emb_lengths = lengths
        # print(embeddings.shape)
        # print(self.embeddings.shape)
        x, _ = self.rnn(x)
        if self.contrastive_on_decoder: 
            embeddings = x 
            emb_lengths = lengths

        # print('after rnn', x.shape)
        self.gru_emb = x.detach()
        x = self.final_decoder(x)
        # print('after final decoder', x.shape)

        return x, lengths, embeddings, emb_lengths

"""
python start_trainer.py --datasetPath /mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release --offset 4 --out_dir CEBRA_cln_AAAA --gru --gauss_in --bidir --batchSize 16 --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 20000 --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder
"""
if __name__=="__main__":

    # import cebra
    # from cebra.models import Offset36Dropoutv2, Offset5Model
    # print(cebra.models.get_options())

    # model = Offset5Model(num_neurons=192, num_units=256, num_output=2048)
    # print(model)
    # B = 64
    # T_max = 4
    # D = 192
    # x = torch.randn((B, T_max, D)).permute(0, 2, 1)
    # lengths = torch.randint(0, T_max, (B, )) + 1 
    # # y, _, embeddings, embedding_l = model(x, lengths)
    # embeddings = model(x).permute(0, 2, 1)
    # print(embeddings.shape)
    # exit()
    ################### the full model 
    model = Encoder_Decoder(
        192,
        64,
        32, # kernel
        4, # stride
        32,
        1024,
        5,
        rnn_bidir=False,
        gru=True, 
        contrastive_on_decoder = True,
        cebra_unfolder=True,
        ceb_hidden=256,
    )
    
    print(model)
    B = 1
    T_max = 1000
    D = 192
    x = torch.randn((B, T_max, D))
    lengths = torch.randint(0, T_max, (B, )) + 1 
    y, _, embeddings, embedding_l = model(x, lengths)
    print(embeddings.shape)

    from torchinfo import summary 
    print(summary(model, input_data=(x,  lengths), verbose=1,))
    
