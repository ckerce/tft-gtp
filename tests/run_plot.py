#!/usr/bin/env python3
"""
Interactive plotting examples - run in Python/Jupyter
"""

import sys
sys.path.insert(0, 'src')

from utils.plotting import *

# Example 2: Compare multiple runs
log_files = [
    "./outputs/wiki_compare/vanilla/training_metrics.json",
    "./outputs/wiki_compare/tft_basic/training_metrics.json",
    "./outputs/wiki_compare/tft_factored/training_metrics.json",
    "./outputs/wiki_dict/tft_dict_factored/training_metrics.json"
]
plot_multiple(log_files)

# Example 3: Perplexity-only comparison
quick_perplexity_plot(log_files)
