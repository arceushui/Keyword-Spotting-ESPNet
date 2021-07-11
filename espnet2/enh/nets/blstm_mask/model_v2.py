#!/usr/bin/env python3

# Copyright (c) 2018 Northwestern Polytechnical University (authors: wujian)
#               2020 National University of Singapore (authors: MA DUO)

import torch

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from nets.blstm_mask.activation import Mish
from nets.blstm_mask.utils import  get_logger

logger = get_logger(__name__)
# from pudb import set_trace
# set_trace()
# compared model.py ,model_v1.py increase num_layers from 3 to 5
class PITNet(torch.nn.Module):
    def __init__(
        self,
        num_bins,
        rnn="lstm",
        num_spks=2,
        num_layers=5,
        hidden_size=896,
        dropout=0.0,
        non_linear="relu",
        bidirectional=True,
    ):
        super(PITNet, self).__init__()
        if non_linear not in ["relu", "sigmoid", "tanh", "mish"]:
            raise ValueError("Unsupported non-linear type:{}".format(non_linear))
        self.num_spks = num_spks
        rnn = rnn.upper()
        if rnn not in ["RNN", "LSTM", "GRU"]:
            raise ValueError("Unsupported rnn type: {}".format(rnn))
        self.rnn = getattr(torch.nn, rnn)(
            num_bins,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.drops = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    hidden_size * 2 if bidirectional else hidden_size, num_bins
                )
                for _ in range(self.num_spks)
            ]
        )
        self.non_linear = {
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.nn.functional.sigmoid,
            "tanh": torch.nn.functional.tanh,
            "mish": Mish(),
        }[non_linear]
        self.num_bins = num_bins

    def forward(self, x, train=True):
        logger.info(f"in network forward input is {x}")
        #logger.info(f"in network forward input shape  is {x.shape}")
        is_packed = isinstance(x, PackedSequence)
        # extend dim when inference
        if not is_packed and x.dim() != 3:
            x = torch.unsqueeze(x, 0)
        x, _ = self.rnn(x)
        # print("====1===")
        #print(f""x)
        # using unpacked sequence
        # x: N x T x D
        if is_packed:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.drops(x)
        # print("====2==")
        # print(x)
        m = []
        for linear in self.linear:
            y = linear(x)
            y = self.non_linear(y)
            if not train:
                y = y.view(-1, self.num_bins)
            m.append(y)
        # print("=====3====")
        # print(m)
        return m

    def disturb(self, std):
        for p in self.parameters():
            noise = torch.zeros_like(p).normal_(0, std)
            p.data.add_(noise)
