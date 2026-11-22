"""MAGMa fragmentation engine, vendored from ms-pred (MIT License).

Only the two files the candidate generator needs are kept: the engine itself
and the subset of ms-pred's chem_utils it uses.
"""
from .fragmentation import FragmentEngine
from .chem_utils import ion2mass
