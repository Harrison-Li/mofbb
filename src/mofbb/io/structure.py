'''
Read atomistic structures using ASE's format support.
'''

from pathlib import Path

import ase
import ase.io


def read_atoms(input_path: str | Path) -> ase.Atoms:
    '''
    Read the last structure frame using ASE.

    Args:
        input_path (str or Path):
            Input file. An @ in the path is treated as part of the filename.

    Returns:
        The last frame as an ase.Atoms object.
    '''
    return ase.io.read(input_path, do_not_split_by_at_sign=True)
