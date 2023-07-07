#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from construct_tff import construct_real_tff

import math
from typing import Optional, List

class TFFLayer():
    def __init__(
        self, 
        k: int, 
        l: int, 
        kmax: int,
        ssss: int,
        n: int, 
        tff_dropout: float,
        merge_weights: bool,
    ):
        self.k = k
        self.l = l
        self.n = n
        if kmax is None:
            self.kmax = k
        else:
            self.kmax = kmax
        if ssss is None:
            self.ssss = 123548997
        else:
            self.ssss = ssss
        # Optional dropout
        if tff_dropout > 0.:
            self.tff_dropout = nn.Dropout(p=tff_dropout)
        else:
            self.tff_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Linear(nn.Linear, TFFLayer):
    # TFF layer implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, # for now outfeatures is also even
        k: int, 
        l: int,  # for now l is an even number only
        kmax: int = None,
        ssss: int = None,
        tff_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        TFFLayer.__init__(self, k=k, l=l, kmax=kmax, ssss=ssss, n=out_features, tff_dropout=tff_dropout,
                           merge_weights=merge_weights)

        print(f'######################### NOTE #######################################33')
        print(f'self.kmax = {self.kmax}, self.k = {self.k}, self.ssss = {self.ssss}')
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if l < out_features:
            generator = torch.Generator().manual_seed(self.ssss)
            ss_indices = torch.randperm(self.k, generator=generator)[:self.kmax]
            tffs = construct_real_tff(self.k, self.l // 2, out_features // 2).permute(0,2,1)[ss_indices,...]
            self.tff_frames = nn.Parameter(torch.cat(tffs.unbind(), dim=1), requires_grad=False)
            self.tff_Ws = nn.Parameter(torch.empty((self.kmax * self.l, in_features)))

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'tff_Ws'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.tff_Ws)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if hasattr(self, 'tff_Ws'):
                    self.weight.data -= T(self.tff_frames @ self.tff_Ws)
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if hasattr(self, 'tff_Ws'):
                    self.weight.data += T(self.tff_frames @ self.tff_Ws)
                    # self.weight.data += T(self.tff_Ws)
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if hasattr(self, 'tff_Ws') and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if hasattr(self, 'tff_Ws'):
                result += (self.tff_dropout(x) @ self.tff_Ws.transpose(0, 1) @ self.tff_frames.transpose(0, 1))
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

# class Embedding(nn.Embedding, LoRALayer):
#     # LoRA implemented in a dense layer
#     def __init__(
#         self,
#         num_embeddings: int,
#         embedding_dim: int,
#         r: int = 0,
#         lora_alpha: int = 1,
#         merge_weights: bool = True,
#         **kwargs
#     ):
#         nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
#                            merge_weights=merge_weights)
#         # Actual trainable parameters
#         if r > 0:
#             self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
#             self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#         self.reset_parameters()
# 
#     def reset_parameters(self):
#         nn.Embedding.reset_parameters(self)
#         if hasattr(self, 'lora_A'):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             nn.init.zeros_(self.lora_A)
#             nn.init.normal_(self.lora_B)
# 
#     def train(self, mode: bool = True):
#         nn.Embedding.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 # Make sure that the weights are not merged
#                 if self.r > 0:
#                     self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 # Merge the weights and mark it
#                 if self.r > 0:
#                     self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
#                 self.merged = True
#         
#     def forward(self, x: torch.Tensor):
#         if self.r > 0 and not self.merged:
#             result = nn.Embedding.forward(self, x)
#             if self.r > 0:
#                 after_A = F.embedding(
#                     x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
#                     self.norm_type, self.scale_grad_by_freq, self.sparse
#                 )
#                 result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
#             return result
#         else:
#             return nn.Embedding.forward(self, x)

            
# class MergedLinear(nn.Linear, LoRALayer):
#     # LoRA implemented in a dense layer
#     def __init__(
#         self, 
#         in_features: int, 
#         out_features: int, 
#         r: int = 0, 
#         lora_alpha: int = 1, 
#         lora_dropout: float = 0.,
#         enable_lora: List[bool] = [False],
#         fan_in_fan_out: bool = False,
#         merge_weights: bool = True,
#         **kwargs
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                            merge_weights=merge_weights)
#         assert out_features % len(enable_lora) == 0, \
#             'The length of enable_lora must divide out_features'
#         self.enable_lora = enable_lora
#         self.fan_in_fan_out = fan_in_fan_out
#         # Actual trainable parameters
#         if r > 0 and any(enable_lora):
#             self.lora_A = nn.Parameter(
#                 self.weight.new_zeros((r * sum(enable_lora), in_features)))
#             self.lora_B = nn.Parameter(
#                 self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
#             ) # weights for Conv1D with groups=sum(enable_lora)
#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#             # Compute the indices
#             self.lora_ind = self.weight.new_zeros(
#                 (out_features, ), dtype=torch.bool
#             ).view(len(enable_lora), -1)
#             self.lora_ind[enable_lora, :] = True
#             self.lora_ind = self.lora_ind.view(-1)
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.transpose(0, 1)
# 
#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'lora_A'):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B)
# 
#     def zero_pad(self, x):
#         result = x.new_zeros((*x.shape[:-1], self.out_features))
#         result = result.view(-1, self.out_features)
#         result[:, self.lora_ind] = x.reshape(
#             -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
#         )
#         return result.view((*x.shape[:-1], self.out_features))
# 
#     def train(self, mode: bool = True):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         nn.Linear.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 # Make sure that the weights are not merged
#                 if self.r > 0 and any(self.enable_lora):
#                     delta_w = F.conv1d(
#                         self.lora_A.data.unsqueeze(0), 
#                         self.lora_B.data.unsqueeze(-1), 
#                         groups=sum(self.enable_lora)
#                     ).squeeze(0)
#                     self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 # Merge the weights and mark it
#                 if self.r > 0 and any(self.enable_lora):
#                     delta_w = F.conv1d(
#                         self.lora_A.data.unsqueeze(0), 
#                         self.lora_B.data.unsqueeze(-1), 
#                         groups=sum(self.enable_lora)
#                     ).squeeze(0)
#                     self.weight.data += self.zero_pad(T(delta_w * self.scaling))
#                 self.merged = True        
# 
#     def forward(self, x: torch.Tensor):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         if self.merged:
#             return F.linear(x, T(self.weight), bias=self.bias)
#         else:
#             result = F.linear(x, T(self.weight), bias=self.bias)
#             if self.r > 0:
#                 after_A = F.linear(self.lora_dropout(x), self.lora_A)
#                 after_B = F.conv1d(
#                     after_A.transpose(-2, -1), 
#                     self.lora_B.unsqueeze(-1), 
#                     groups=sum(self.enable_lora)
#                 ).transpose(-2, -1)
#                 result += self.zero_pad(after_B) * self.scaling
#             return result
#         
# 
# class ConvLoRA(nn.Module, LoRALayer):
#     def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
#         super(ConvLoRA, self).__init__()
#         self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
#         assert isinstance(kernel_size, int)
#         # Actual trainable parameters
#         if r > 0:
#             self.lora_A = nn.Parameter(
#                 self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
#             )
#             self.lora_B = nn.Parameter(
#               self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
#             )
#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.conv.weight.requires_grad = False
#         self.reset_parameters()
#         self.merged = False
# 
#     def reset_parameters(self):
#         self.conv.reset_parameters()
#         if hasattr(self, 'lora_A'):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B)
# 
#     def train(self, mode=True):
#         super(ConvLoRA, self).train(mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 # Make sure that the weights are not merged
#                 self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 # Merge the weights and mark it
#                 self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
#                 self.merged = True
# 
#     def forward(self, x):
#         if self.r > 0 and not self.merged:
#             return self.conv._conv_forward(
#                 x, 
#                 self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
#                 self.conv.bias
#             )
#         return self.conv(x)
# 
# class Conv2d(ConvLoRA):
#     def __init__(self, *args, **kwargs):
#         super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)
# 
# class Conv1d(ConvLoRA):
#     def __init__(self, *args, **kwargs):
#         super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)
# 
# # Can Extend to other ones like this
# 
# class Conv3d(ConvLoRA):
#     def __init__(self, *args, **kwargs):
#         super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)
