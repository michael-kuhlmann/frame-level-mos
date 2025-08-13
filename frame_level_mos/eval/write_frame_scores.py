from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from operator import itemgetter
from pathlib import Path
import pickle

try:
    import dlp_mpi
    DLP_MPI_AVAILABLE = True
except ImportError:
    DLP_MPI_AVAILABLE = False

import click
import numpy as np
from paderbox.io import load_audio, load_json
import padertorch as pt
from padertorch.utils import to_numpy
import psutil
import torch
from tqdm import tqdm
from sed_scores_eval.base_modules.io import write_sed_scores

from ..lazy_dataset_utils import from_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def map_audio(example):
    audio_path = example['file_path']
    audio, sr = load_audio(audio_path, return_sample_rate=True, dtype='float32')
    example['audio'] = audio
    example['sampling_rate'] = sr
    example['num_samples'] = audio.shape[-1]
    example['example_id'] = audio_path.stem
    return example


def pad_audio(example):
    example['audio'] = pt.pad_sequence(example['audio'], batch_first=True)
    return example


def prepare_dataset(dataset):
    dataset = dataset.map(map_audio)
    return dataset


def worker_fn(
    batch,
    scores_dir,
    model,
    ground_truth,
    perturbations,
    no_clamp=False,
    clamp_min=1.,
    clamp_max=5.,
    resume=False,
):
    example_ids = batch["example_id"]
    if (
        len(example_ids) == 1
        and (scores_dir / f"{example_ids[0]}.tsv").exists()
        and resume
    ):
        return
    audio = batch["audio"]
    if audio.ndim == 2:
        audio = audio.unsqueeze(1)
    _, preds, seq_len_preds = model(
        audio, batch["num_samples"],
    )
    if no_clamp:
        preds = -preds.squeeze(1) # (B, T)
    else:
        # Normalize to [0, 1]
        preds = preds.clamp(clamp_min, clamp_max)
        preds = (
            (preds - clamp_min)
            / (clamp_max - clamp_min)
        )
        # Invert
        preds =  1 - preds.squeeze(1)  # (B, T)

    # Create and save scores dataframe
    for pred, seq_len, example_id, num_samples, sr in zip(
        preds,
        seq_len_preds,
        example_ids,
        batch["num_samples"],
        batch["sampling_rate"],
    ):
        if (scores_dir / f"{example_id}.tsv").exists() and resume:
            continue
        pred = to_numpy(pred[:seq_len])
        timestamps = np.linspace(
            0, num_samples/sr,
            num=seq_len+1, endpoint=True,
        )
        # Separate scores for each perturbation class
        intervals = ground_truth["data"][example_id]
        active = list(
            map(perturbations.index, map(itemgetter(2), intervals))
        )
        unq_active = np.unique(active)
        _scores = np.zeros(
            (pred.shape[0], len(perturbations)), dtype=pred.dtype
        )
        for ind in unq_active.astype(int):
            _scores[:, ind] = pred
        event_classes = perturbations
        write_sed_scores(
            _scores,
            scores_dir / f"{example_id}.tsv",
            timestamps=timestamps,
            event_classes=event_classes,
        )


@click.command()
@click.argument(
    'input_dir', type=click.Path(exists=True, file_okay=False),
)
@click.argument(
    'model_dir', type=click.Path(exists=True, file_okay=False),
)
@click.option(
    '--checkpoint-name', type=str, default='ckpt_best_SRCC.pth',
    help="Name of the checkpoint to load from the model directory.",
)
@click.option(
    '--no-weights-only', is_flag=True,
    help=(
        'Uses torch.load(..., weights_only=False) to load the model. '
        'Use this if you trust the checkpoint and run into an UnpicklingError '
        'when loading the model.'
    ),
)
@click.option(
    '--no-clamp', is_flag=True,
    help='Do not clamp the scores to the range [clamp_min, clamp_max].',
)
@click.option(
    '--clamp_min', type=float, default=1.,
)
@click.option(
    '--clamp_max', type=float, default=5.,
)
@click.option(
    '--resume', is_flag=True,
    help='Do not overwrite existing scores.',
)
@click.option(
    '--backend', type=click.Choice(['t', 'dlp_mpi']), default='t',
    help=(
        'Backend to use for parallel processing. '
        'Possible choices are threads ("t") and MPI ("dlp_mpi"). '
        'Defaults to "t" (threads).'
    ),
)
@click.option(
    '--num-workers', '-j', type=int, default=-1,
    help=(
        'Number of workers to use for parallel processing. '
        'If -1, uses all available CPU cores.'
    ),
)
def main(
    input_dir,
    model_dir,
    checkpoint_name: str = 'ckpt_best_SRCC.pth',
    no_weights_only: bool = False,
    no_clamp: bool = False,
    clamp_min: float = 1.,
    clamp_max: float = 5.,
    resume: bool = False,
    backend: str = 't',
    num_workers: int = -1,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    input_dir = Path(input_dir)
    model_dir = Path(model_dir)

    scores_dir = model_dir / input_dir.name / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    done_file = scores_dir.parent / ".done"

    ground_truth = load_json(input_dir / "ground_truth.json")
    perturbations = ground_truth["meta"]["perturbations"]

    if not done_file.exists():
        try:
            model = pt.Module.from_storage_dir(
                model_dir, config_name="config.yaml",
                checkpoint_name=checkpoint_name,
                consider_mpi=backend=='dlp_mpi',
                weights_only=not no_weights_only,
            )
        except pickle.UnpicklingError as exc:
            logger.error(
                "UnpicklingError while loading the model. "
                "Try using the --no-weights-only flag to load the model."
            )
            raise exc
        model = model.to(device).eval()

        dataset = from_path(input_dir, suffix=".wav")
        len_dataset = len(dataset)
        if not hasattr(model, "prepare_example"):
            model.prepare_example = lambda x: x
        dataset = (
            prepare_dataset(dataset)
            .map(model.prepare_example)
            .batch(1)
            .map(pt.data.utils.collate_fn)
            .map(partial(pt.data.example_to_device, device=device))
            .map(pad_audio)
        )

        _worker_fn = partial(
            worker_fn,
            scores_dir=scores_dir,
            model=model,
            ground_truth=ground_truth,
            perturbations=perturbations,
            no_clamp=no_clamp,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            resume=resume,
        )
        if backend == 't':
            if num_workers < 0:
                num_workers = len(psutil.Process().cpu_affinity())
            logger.info("Using threads backend with %d workers", num_workers)
            executor = tqdm(ThreadPoolExecutor(max_workers=num_workers).map(
                _worker_fn, dataset,
            ), total=len_dataset)
        else:
            if not DLP_MPI_AVAILABLE:
                raise ImportError(
                    "dlp_mpi is not available. "
                    "Please install it to use the 'dlp_mpi' backend.\n"
                    "See: https://github.com/fgnt/dlp_mpi"
                )
            logger.info("Using MPI backend")
            executor = dlp_mpi.map_unordered(
                _worker_fn, dataset,
                indexable=False,
                progress_bar=True,
            )
        for _ in executor:
            pass
    else:
        logger.info(
            "Scores already written. "
            "If you want to overwrite them, delete the file %s",
            done_file,
        )
        return

    if backend == 't' or (DLP_MPI_AVAILABLE and dlp_mpi.IS_MASTER):
        done_file.touch()
        logger.info(f"Scores written to {scores_dir}")
        # Symlink ground truth and durations files next to scores directory
        if not (scores_dir.parent / "ground_truth.json").exists():
            (scores_dir.parent / "ground_truth.json").symlink_to(
                input_dir / "ground_truth.json"
            )
        if not (scores_dir.parent / "audio_durations.json").exists():
            (scores_dir.parent / "audio_durations.json").symlink_to(
                input_dir / "audio_durations.json"
            )

        total_areas = 0
        area_durations = []
        for _, timestamps in ground_truth["data"].items():
            for timestamp in timestamps:
                total_areas += 1
                area_durations.append(timestamp[1]-timestamp[0])
        logger.info(
            "### Statistics ###",
        )
        logger.info(
            "Average number of distortions per utterance: %.2f",
            total_areas/len(ground_truth["data"]),
        )
        logger.info(
            "Average duration of distortions: %.3f±%.3f\n",
            np.mean(area_durations), np.std(area_durations),
        )


if __name__ == '__main__':
    main()
