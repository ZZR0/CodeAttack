"""
Shared TextAttack Functions
=============================

This package includes functions shared across packages.

"""


from . import data
from . import utils
from .utils import logger
from . import validators

from .attacked_code import AttackedCode
from .word_embeddings import AbstractWordEmbedding, WordEmbedding, GensimWordEmbedding
from .checkpoint import AttackCheckpoint
