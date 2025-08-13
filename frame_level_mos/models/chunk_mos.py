import typing as tp

import numpy as np
import padertorch as pt
from padertorch.contrib.mk.typing import TSeqLen
from padertorch.data.segment import segment_axis
from padertorch.utils import to_numpy
import torch
from torch import Tensor
import torch.nn.functional as F

from .ssl_mos import SSLMOS


class ChunkMOS(SSLMOS):
    def __init__(
        self,
        *args,
        segment_length: tp.Optional[int] = None,
        min_segment_length: tp.Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.segment_length = segment_length
        self.min_segment_length = min_segment_length
        if self.min_segment_length is None:
            self.min_segment_length = segment_length

    def _segment_single_audio(
        self,
        audio: Tensor,
        length: tp.Optional[int],
        num_samples: tp.Optional[int] = None,
        segment_boundaries: tp.Optional[Tensor] = None,
    ):
        if num_samples is not None:
            audio = audio[..., :num_samples]
        else:
            num_samples = audio.shape[-1]
        if segment_boundaries is not None:
            segment_boundaries = segment_boundaries.long()
            segment_boundaries[-1] = num_samples
            segment_lengths = np.diff(to_numpy(segment_boundaries))
            segment_lengths = np.r_[
                to_numpy(segment_boundaries)[0], segment_lengths
            ]
            boundaries = segment_lengths.cumsum(0)[:-1]
            segments = torch.split(
                audio.moveaxis(-1, 0),
                segment_lengths.tolist(),
                dim=0
            )
            segments = pt.pad_sequence(segments, batch_first=True)
            if audio.ndim > 1:
                segments = segments.moveaxis(-1, 0)
            return segments, segment_lengths, boundaries

        if length is None:
            return audio.unsqueeze(-2), [num_samples]

        segments = segment_axis(
            audio, length, length, axis=-1, end='pad'
        )
        segment_lengths = np.array([length]*segments.shape[-2])
        if num_samples is not None and num_samples % (length) != 0:
            segment_lengths[-1] = num_samples % (length)
        if segment_lengths[-1] < self.min_segment_length:
            if segments.ndim == 3:
                segments = segments.squeeze(0)
            segments = list(segments)
            last = segments.pop()
            segments[-1] = torch.cat(
                (segments[-1], last[..., :segment_lengths[-1]]), dim=-1
            )
            segments = pt.pad_sequence(segments, batch_first=True)
            if audio.ndim == 2:
                segments = segments.unsqueeze(0)
            elif audio.ndim != 1:
                raise ValueError(
                    f"Expected audio to have 1 or 2 dimensions, "
                    f"got {audio.ndim} dimensions."
                )
            segment_lengths = list(segment_lengths)
            last_length = segment_lengths.pop()
            segment_lengths[-1] += last_length
            segment_lengths = np.array(segment_lengths)
        boundaries = np.cumsum(segment_lengths)[:-1]
        assert_sizes = segment_lengths
        assert sum(assert_sizes) == num_samples, (
            sum(assert_sizes), num_samples,
            length, segments.shape[-2]
        )
        return segments, segment_lengths, boundaries

    def _stitch_single_audio(
        self,
        segments: Tensor,
        segment_lengths: TSeqLen,
        target_seq_len,
    ):
        sequence = torch.cat(pt.unpad_sequence(
            segments.moveaxis(1, 0), segment_lengths
        ))
        return sequence, target_seq_len

    def segment(
        self,
        audio: Tensor,
        lengths: tp.Optional[tp.Union[int, tp.List[int]]],
        num_samples: TSeqLen = None,
        segment_boundaries=None,
    ):
        if num_samples is None:
            num_samples = [None]*len(audio)
        if isinstance(lengths, int):
            lengths = [lengths]*len(audio)
        elif lengths is None:
            lengths = [None]*len(audio)
        if segment_boundaries is None:
            segment_boundaries = [None]*len(audio)
        segment_results = list(map(
            self._segment_single_audio, audio,
            lengths, num_samples, segment_boundaries,
        ))
        segments, segment_lengths, boundaries = zip(
            *segment_results
        )
        segments_per_audio = list(map(lambda t: t.shape[-2], segments))
        segment_lengths = torch.from_numpy(np.concatenate(segment_lengths))
        pad_len = max(segment_lengths)
        segments = list(map(
            lambda t: F.pad(t, (0, pad_len-t.shape[-1])), segments
        ))
        segments = torch.cat(segments, dim=-2).moveaxis(-2, 0)
        return (
            segments, segment_lengths, segments_per_audio, boundaries
        )

    def stitch(
        self,
        segments: Tensor,
        segment_lengths: TSeqLen,
        segments_per_audio: TSeqLen,
        target_sequence_lengths: TSeqLen,
    ):
        if segments_per_audio is None:
            return segments, segment_lengths
        segments = torch.split(segments, segments_per_audio)
        segment_lengths = np.split(
            segment_lengths, np.cumsum(segments_per_audio)
        )
        sequences, seq_lens = zip(*map(
            self._stitch_single_audio,
            segments,
            segment_lengths,
            target_sequence_lengths,
        ))
        batch = pt.pad_sequence(sequences, batch_first=True)
        return batch, np.asarray(seq_lens)

    def forward(
        self,
        audio: Tensor,
        num_samples=None,
        average: bool = True,
        return_raw: bool = False,
        return_latents: bool = False,
        segment_boundaries=None,
    ):
        with torch.no_grad():
            segments, segment_lengths, segs_per_wav, _ = self.segment(
                audio, self.segment_length, num_samples,
                segment_boundaries=segment_boundaries,
            )
            xs, seq_len_xs = self.encode(segments, segment_lengths)
            target_sequence_lengths = self.encoder.compute_output_lengths(
                num_samples
            )
            x, seq_len_x = self.stitch(
                xs, seq_len_xs,
                segs_per_wav,
                target_sequence_lengths,
            )
            if isinstance(seq_len_x, np.ndarray):
                seq_len_x = seq_len_x.tolist()
            y, seq_len_x = self.transform(x, seq_len_x)
            preds, frame_preds = self.aggregate(y, seq_len_x)
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
