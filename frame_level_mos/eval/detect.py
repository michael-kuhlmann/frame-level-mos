import logging
from pathlib import Path
import typing as tp

import click
import numpy as np
from paderbox.io import load_json, dump
import psutil
from sed_scores_eval.base_modules.io import parse_scores
from sed_scores_eval.intersection_based import pipsds

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def format_dict(
    d: dict,
    macro_average=True,
    micro_average=False,
    precision: int = 3,
) -> str:
    s = ""
    if macro_average:
        s += f"{d.pop('macro_average'):.{precision}f}"
    else:
        d.pop('macro_average', None)
    if micro_average:
        if macro_average:
            raise ValueError(
                "macro_average and micro_average cannot be both True."
            )
        s += f"{d.pop('micro_average')}:.{precision}f"
    else:
        d.pop('micro_average', None)
    s += (
        "".join((f"\n\t{k}: {v:.{precision}f}" for k, v in d.items()))
    )
    return s


@click.command()
@click.argument(
    'scores_dir', type=click.Path(exists=True, file_okay=False),
)
@click.option(
    '--min-filter-length', type=float, default=0.,
    help='Minimum filter length in seconds for median filtering.',
)
@click.option(
    '--max-filter-length', type=float, default=0.5,
    help='Maximum filter length in seconds for median filtering.',
)
@click.option(
    '--num-filters', type=int, default=11,
    help='Number of filter lengths to use for median filtering.',
)
@click.option(
    '--dtc-threshold', type=float, default=0.5,
    help='DTC threshold for AUC calculation.',
)
@click.option(
    '--gtc-threshold', type=float, default=None,
    help=(
        'GTC threshold for AUC calculation. If not provided, defaults to ' 
        'dtc-threshold.'
    ),
)
@click.option(
    '--max-efpr', type=float, default=100.,
    help=(
        'Maximum effective false positive rate for AUC calculation. '
        'Defaults to 100.'
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
    scores_dir,
    min_filter_length: float = 0.,
    max_filter_length: float = 0.5,
    num_filters: int = 11,
    dtc_threshold: float = 0.5,
    gtc_threshold: tp.Optional[float] = None,
    max_efpr: float = 100.,
    num_workers: int = -1,
):
    scores_dir = Path(scores_dir)

    ground_truth = load_json(scores_dir / "ground_truth.json")
    durations = load_json(scores_dir / "audio_durations.json")
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

    scores, _ = parse_scores(scores_dir / 'scores')
    median_filter_lengths_in_sec = np.linspace(
        min_filter_length,
        max_filter_length,
        num_filters,
    )
    if gtc_threshold is None:
        gtc_threshold = dtc_threshold
    if num_workers < 0:
        num_workers = len(psutil.Process().cpu_affinity())
    logger.info("Number of workers: %d", num_workers)

    pi_auc, classwise_pi_auc, pi_roc, *_ =\
        pipsds.median_filter_independent_psds(
            scores,
            ground_truth["data"],
            durations,
            median_filter_lengths_in_sec=median_filter_lengths_in_sec,
            dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
            max_efpr=max_efpr,
            num_jobs=num_workers,
        )
    classwise_pi_auc["macro_average"] = pi_auc
    logger.info(
        "### Intersection-based scores (DTC=%.1f, GTC=%.1f) ###",
        dtc_threshold, gtc_threshold,
    )
    logger.info(
        "PI-AUC: %s\n", format_dict(classwise_pi_auc),
    )
    dump(
        {"tpr": pi_roc[0], "fpr": pi_roc[1]},
        scores_dir / f"roc_dtc_{dtc_threshold}_gtc_{gtc_threshold}.json"
    )


if __name__ == '__main__':
    main()
