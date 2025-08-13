import io
from pathlib import Path
import typing as tp

import numpy as np
from paderbox.io import dumps_audio
from soundfile import SoundFile


def normalize_loudness(
    audio: tp.Union[str, Path, np.ndarray],
    target_dBFS: float = -18.0,
    *,
    sampling_rate: tp.Optional[int] = None,
    output_file: tp.Optional[tp.Union[str, Path]] = None,
    exist_ok: bool = False,
    skip_exist: bool = False,
):
    """Normalize the average loudness of the input audio.

    Args:
        audio (Union[str, Path], np.ndarray): Input audio file or array.
        output_file (Optional[Union[str, Path]], optional): Output audio file.
            If None, will create the output file in the same folder as the
            input file and append the suffix "_norm". Defaults to None.
        target_dBFS (float, optional): Target loudness in dBFS. Defaults to
            -18.0 which is the recommended alignment level for Europe by the
            European Broadcasting Union (EBU).
        exist_ok (bool, optional): If True, overwrite the existing output file.
            If False and the output file exists, raise an error. Defaults to
            False.
        skip_exist (bool, optional): If True, skip normalization if the output
            file exists. Defaults to False.
    """
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise ImportError(
            "Please install pydub to use this function (normalize_loudness). "
            "You can install it via pip: pip install pydub"
        ) from e

    def _normalize_loudness(_sound: AudioSegment):
        # dBFS measures the ratio between the audio's RMS level (average loudness) and its maximum possible amplitude
        # A target dBFS of 0 dBFS means that average loudness is equal to
        # the maximum possible amplitude
        change_in_dBFS = target_dBFS - _sound.dBFS
        return _sound.apply_gain(change_in_dBFS)

    if isinstance(audio, (str, Path)):
        file = Path(audio)
        if output_file is None:
            output_file = (file.parent / (file.stem + "_norm")).with_suffix(".wav")
        output_file = Path(output_file)
        if output_file.exists():
            if not exist_ok:
                raise FileExistsError(output_file)
            if skip_exist:
                return
        sound = AudioSegment.from_file(file, "wav")
        x = _normalize_loudness(sound)
        x.export(output_file, format="wav")
        return

    dtype = audio.dtype
    data = dumps_audio(audio, sample_rate=sampling_rate, normalize=False)
    sound = AudioSegment(data=data)
    sound = _normalize_loudness(sound)
    with io.BytesIO() as f:
        sound.export(f, format="wav")
        view = f.read()
    with io.BytesIO(view) as f:
        sf = SoundFile(f)
        array = sf.read(dtype=dtype)
    while array.ndim < audio.ndim:
        array = np.expand_dims(array, axis=0)
    return array
