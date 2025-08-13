# Towards Frame-level Quality Predictions of Synthetic Speech

We provide a small toolkit to evaluate frame-level representations of MOS predictors.
The task is to detect local distortions that were inserted into clean speech recordings.
We use the intersection-based detection criterion [[1](#1), [2](#2)] from the sound event detection community for evaluation.

## Installation
### Via pip
```bash
pip install git+https://github.com/fgnt/frame-level-mos.git
```

### From source
```bash
git clone https://github.com/fgnt/frame-level-mos.git
cd frame-level-mos
pip install -e .
```

## Data

You can download two modified LibriTTS-R datasets with inserted local distortions from huggingface:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="kuhlmannm/is25-local-distortions", local_dir="is25_local_distortions", repo_type="dataset", ignore_ptterns=["*.txt"])
```

## Checkpoints

You can download the following checkpoints from huggingface:
```python
from huggingface_hub import snapshot_download
```

|Model|repo id|Command|
|:----|:------|:------|
|SHEET SSL-MOS: Re+CNN|`kuhlmannm/is25-ssl-mos-cnn`|```snapshot_download(repo_id="kuhlmannm/is25-ssl-mos-cnn", local_dir="ssl_mos_cnn")```|
|ChunkMOS+BLSTM|`kuhlmannm/is25-chunk-mos-blstm`|```snapshot_download(repo_id="kuhlmannm/is25-chunk-mos-blstm", local_dir="chunk_mos_blstm")```|

## How to use

The evaluation consists of two steps:
1. Save the frame-level scores for each speech signal to evaluate to a single file.
2. Compute the detection score by evaluating the frame-level scores against the ground truth.

### Using the provided checkpoints and data

1. Save the frame-level scores for each speech signal to evaluate to a single file.
```bash
python -m frame_level_mos.eval.write_frame_scores /path/to/downloaded/data/<dataset_name> /path/to/downloaded/model
```
The frame-level scores will be saved under `/path/to/downloaded/model/<dataset_name>/scores`.
See `python -m frame_level_mos.eval.write_frame_scores --help` for more options.

2. Compute the detection score by evaluating the frame-level scores against the ground truth.
```bash
python -m frame_level_mos.eval.detect /path/to/downloaded/model/<dataset_name>
```
This will print the intersection-based detection scores.
By default, an intersection threshold of 0.5 is used.
To use a different threshold, you can use the `--dtc-threshold` option.
See `python -m frame_level_mos.eval.detect --help` for more options.

#### Results

##### dev_clean

|Model|DTC=GTC=0.5|DTC=GTC=0.7|
|:----|:--------:|:--------:|
|SSL-MOS [[3]](#3)|.693|.555|
|SHEET SSL-MOS: Re+CNN|**.748**|**.634**|
|ChunkMOS+BLSTM|.638|.337|

##### test_clean

|Model|DTC=GTC=0.5|DTC=GTC=0.7|
|:----|:--------:|:--------:|
|SSL-MOS [[3]](#3)|.703|.562|
|SHEET SSL-MOS: Re+CNN|**.748**|**.625**|
|ChunkMOS+BLSTM|.647|.341|

### Using your own data

If you would like to use your own data, please use the following structure:
```
/data/root
|-- ground_truth.json
|-- audio_durations.json
|-- audio_files/
    |-- audio1.wav
    |-- audio2.wav
    |-- ...
```
The audio files should be in WAV format.
It is also possible to arrange the audio files in subdirectories.

The file `ground_truth.json` contains the start and stop of each distortion in the following format:
```python
{
  "data": {
    "audio1": [(0.475, 1.105, "class2"), (2.42, 2.83, "class3")],
    "audio2": [(3.49, 3.89, "class1"), (6.16, 6.56, "class2"), (7.11, 7.51, "class1")],
    ...,
    "meta": {
        "perturbations": [
            "class1",
            "class2",
            "class3",
            ...
        ]
    }
  }
}
```

The file `audio_durations.json` contains the duration of each audio file in seconds:
```python
{
  "data": {
    "audio1": 5.04,
    "audio2": 6.10,
    ...
  }
}
```

### Using your own model

#### Option 1: Write a wrapper for padertorch
Write a wrapper that inherits from [padertorch.Module](https://github.com/fgnt/padertorch/blob/5d00ac5cfde54fab7bdacb31a8b513ee728a2aa7/padertorch/base.py#L55).
Its `forward` method should take a batch of audio signals and their sequence lengths as input and return the utterance-level predictions, frame-level scores, and the sequence lengths of the frame-level scores.

#### Option 2: Save the frame-level scores yourself
Instead of using [write_scores_frame.py](frame_level_mos/eval/write_frame_scores.py), you can save the frame-level scores yourself.
See [sed_scores_eval](https://github.com/fgnt/sed_scores_eval) for how to do this.

## References
<a id="1">[1]</a> C. Bilen, G. Ferroni, F. Tuveri, J. Azcarreta and S. Krstulovic,
"A Framework for the Robust Evaluation of Sound Event Detection",
in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
2020.
[Paper Link.](https://ieeexplore.ieee.org/abstract/document/9052995)

<a id="2">[2]</a> J. Ebbers, R. Serizel and R. Haeb-Umbach,
"Threshold-Independent Evaluation of Sound Event Detection Scores",
in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2022.
[Paper Link.](https://ieeexplore.ieee.org/abstract/document/9747556)

<a id="3">[3]</a> [mos-finetune-ssl](https://github.com/nii-yamagishilab/mos-finetune-ssl/blob/main/run_inference.py)

## Citation
```bibtex
@inproceedings{kuhlmann25_interspeech,
  title     = {{Towards Frame-level Quality Predictions of Synthetic Speech}},
  author    = {{Michael Kuhlmann and Fritz Seebauer and Petra Wagner and Reinhold Haeb-Umbach}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
}
```

## Acknowledgements
This research was funded by Deutsche Forschungsgemeinschaft (DFG), project 446378607.
