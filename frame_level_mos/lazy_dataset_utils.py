from pathlib import Path
import typing as tp

import lazy_dataset


def from_path(
    root: tp.Union[str, Path],
    suffix: str,
    immutable_warranty: str = 'pickle',
    name: tp.Optional[str] = None,
):
    import os
    # https://stackoverflow.com/a/59803793/16085876
    def run_fast_scandir(root: Path, ext: tp.List[str]):
        subfolders, files = [], []

        for f in os.scandir(root):
            if f.is_dir():
                subfolders.append(f.path)
            if f.is_file():
                # if os.path.splitext(f.name)[1].lower() in ext:
                if any(e in f.name.lower() for e in ext):
                    files.append(Path(f.path))

        for folder in list(subfolders):
            sf, f = run_fast_scandir(folder, ext)
            subfolders.extend(sf)
            files.extend(f)
        return subfolders, files

    _, files = run_fast_scandir(Path(root), [suffix])
    files = map(Path, files)
    examples = {file.stem: {"file_path": file} for file in files}
    return lazy_dataset.from_dict(examples, immutable_warranty, name)
