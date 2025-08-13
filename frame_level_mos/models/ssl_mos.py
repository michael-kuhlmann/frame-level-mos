from functools import partial
import typing as tp

import numpy as np
import padertorch as pt
from padertorch.contrib.je.modules.reduce import Mean
from padertorch.contrib.mk.typing import TSeqLen
from padertorch.ops.mappings import ACTIVATION_FN_MAP
import torch
from torch import Tensor, nn

from ..utils import normalize_loudness


class SSLMOS(pt.Module):
    def __init__(
        self,
        encoder: pt.Module,
        conv_net: tp.Optional[pt.Module] = None,
        bilstm: tp.Optional[pt.Module] = None,
        d_model: int = 768,
        num_layers: int = 12,
        out_activation=None,
        scale: float = 1.,
        bias: float = 0.,
        normalize_ratings: bool = False,
        standardize_audio: bool = False,
        equal_loudness: bool = True,
        l2_normalization: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.conv_net = conv_net
        self.bilstm = bilstm
        if out_activation is not None:
            self.out_activation = ACTIVATION_FN_MAP[out_activation]()
        else:
            self.out_activation = nn.Identity()

        if bilstm is not None:
            assert bilstm.bidirectional
            proj_size = 2*bilstm.hidden_size
        elif conv_net is not None:
            proj_size = conv_net.out_channels[-1]
        else:
            proj_size = d_model
        self.out_proj = nn.Linear(proj_size, 1)
        self.weights = torch.nn.Parameter(
            torch.ones(num_layers), requires_grad=True
        )

        self.scale = scale
        self.bias = bias
        self.normalize_ratings = normalize_ratings
        self.standardize_audio = standardize_audio
        self.equal_loudness = equal_loudness
        self.l2_normalization = l2_normalization

    def _normalize_ratings(self, ratings):
        return (ratings - 1) / 2 - 1  # [-1, 1]

    def zero_init_(self):
        for param in self.out_proj.parameters():
            param.detach().zero_()

    def inverse_normalization(self, scores: tp.Union[Tensor, np.ndarray]):
        if not self.normalize_ratings:
            return scores
        return (scores + 1) * 2 + 1  # [1, 5]

    def prepare_example(self, example):
        observation = example["audio"]
        if self.equal_loudness:
            if observation.ndim > 2:
                observation = np.stack(list(map(
                    partial(
                        normalize_loudness,
                        sampling_rate=example["sampling_rate"]
                    ), observation,
                )))
            else:
                observation = normalize_loudness(
                    observation, sampling_rate=example["sampling_rate"]
                )
        if self.standardize_audio:
            observation = (
                (observation - np.mean(observation, axis=-1, keepdims=True))
                / (np.std(observation, axis=-1, keepdims=True) + 1e-7)
            )
        example["audio"] = observation
        return example

    def flatten_encoder_output(
        self, x: tp.Union[Tensor, tp.List[Tensor]], seq_len_x: TSeqLen = None,
    ):
        if isinstance(x, list):
            # Weighted sum of layers
            x = torch.stack(x, dim=-1)
            w = torch.relu(self.weights)
            w = w / w.sum()
            while w.ndim < x.ndim:
                w = w.unsqueeze(0)
            x = (x * w).sum(dim=-1)
        if self.l2_normalization:
            m = Mean(axis=1, keepdims=True)(x, seq_len_x)
            norm = torch.linalg.norm(m, ord=2, dim=-1, keepdim=True)
            x = x / (norm + 1e-6)
        return x

    def transform(self, x, seq_len_x):
        if self.conv_net is not None:
            x, seq_len_x = self.conv_net(
                x.mT, seq_len_x
            )
            x = x.mT
        if self.bilstm is not None:
            x = pt.pack_padded_sequence(
                x, seq_len_x, batch_first=True,
                enforce_sorted=True,
            )
            x, _ = self.bilstm(x)
            x, _ = pt.pad_packed_sequence(x, batch_first=True)
        return x, seq_len_x

    def reduce(
        self, x: Tensor, seq_len_x: TSeqLen,
    ):
        return Mean(axis=1)(x, seq_len_x)

    def aggregate(self, x: Tensor, seq_len_x: TSeqLen):
        preds = self.out_proj(x)
        preds = self.out_activation(preds).squeeze(-1)
        preds = preds*self.scale+self.bias
        return self.reduce(preds, seq_len_x), preds

    def encode(
        self, wavs: Tensor, num_samples: TSeqLen,
    ):
        latents, seq_len_x = self.encoder(
            wavs, num_samples, return_latents=True
        )
        x = self.encoder.extract_features_from_latents(
            latents, seq_len_x
        )
        x = self.flatten_encoder_output(x, seq_len_x)
        return x, seq_len_x

    def forward(
        self,
        audio: Tensor,
        num_samples=None,
        average: bool = True,
        return_raw: bool = False,
        return_latents: bool = False,
    ):
        with torch.no_grad():
            x, seq_len_x = self.encode(
                audio, num_samples,
            )
            if isinstance(seq_len_x, np.ndarray):
                seq_len_x = seq_len_x.tolist()
            y, seq_len_y = self.transform(x, seq_len_x)
            preds, frame_preds = self.aggregate(y, seq_len_y)
            frame_scores = self.out_proj(y).squeeze(-1)
            if not return_raw:
                frame_scores = self.inverse_normalization(frame_preds)
            outputs = [frame_scores, seq_len_x]
            if average:
                preds = self.inverse_normalization(preds)
                outputs.insert(0, preds)
            if return_latents:
                outputs.append((x, y))
            return tuple(outputs)
