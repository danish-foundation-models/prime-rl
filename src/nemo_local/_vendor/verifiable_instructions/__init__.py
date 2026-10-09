"""
Verifiable Instructions - Lightweight verifiable instructions for training.

This package provides instruction classes and synthesis tools for generating
instruction following datasets using verifiable training constraints.
"""

from .instructions_registry import INSTRUCTION_DICT, INSTRUCTION_CONFLICTS

__version__ = "0.1.0"
__all__ = ["INSTRUCTION_DICT", "INSTRUCTION_CONFLICTS"]
