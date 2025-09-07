from typing import Union

from .dataset import CustomDataset
from pathlib import Path

import random
import os

DEBUG = os.environ["DEBUG"]

def create_datasets(path: Union[str, Path]) -> list[CustomDataset]:
    """Create train/validation/test datasets from a folder of classed images.

    Expects ``path`` to contain three subfolders named ``NORMAL``,
    ``PNEUMONIA``, and ``COVID`` with ``.png`` images inside each. The
    function performs a simple per-class 80/10/10 split by slicing the list
    of files in that order (no shuffling) and returns three ``CustomDataset``
    instances in the order: train, validation, test.

    Labels are mapped as: ``NORMAL -> 0``, ``PNEUMONIA -> 1``, ``COVID -> 2``.
    A fixed random seed is set for determinism, though the current split does
    not shuffle file lists.

    Args:
        path: Root directory containing the class subfolders with ``.png`` images.

    Returns:
        A list ``[train_set, validation_set, test_set]`` where each element is a
        ``CustomDataset`` built from ``(image_path, label_index)`` pairs.
    """
    random.seed(7)
    data_path = Path(path)

    classes_to_idx = {
        'NORMAL': 0,
        'PNEUMONIA': 1,
        'COVID': 2
    }

    classes = classes_to_idx.keys()

    train_set = []
    validation_set = []
    test_set = []

    for c in classes:
        path_and_idx = [(f,classes_to_idx[c])  for f in Path(data_path / c).glob('*.png')]

        n = len(path_and_idx)
        n_train, n_val = int(0.8*n), int(0.1*n)

        train_set.extend(path_and_idx[:n_train])
        validation_set.extend(path_and_idx[n_train: n_train + n_val])
        test_set.extend(path_and_idx[n_train + n_val:])


    if DEBUG:
        n = int(.1*len(train_set))
        random.shuffle(train_set)
        train_set = train_set[:n]

    return [
        CustomDataset(arr) for arr in [
            train_set, validation_set, test_set
        ]
    ]

if __name__ == "__main__":
    train, _, _ = create_datasets('./dataset')