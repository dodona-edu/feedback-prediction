#!/usr/bin/env python3
"""
Module for preparing an so file for analysis.
"""

from tree_sitter import Language

Language.build_library(
    'build/languages.so',
    [
        'tree-sitter-python'
    ]
)
