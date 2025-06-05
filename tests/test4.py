import sys
sys.path.insert(0, 'src')
# 2. Compare multiple runs
from utils.plotting2 import plot_training_curves
log_files = [
    "outputs/wiki_compare/vanilla/training_metrics.json",
    "outputs/wiki_compare/tft_basic/training_metrics.json", 
    "outputs/wiki_compare/tft_factored/training_metrics.json",
    "outputs/wiki_dict/tft_dict_factored/training_metrics.json"
]
plot_training_curves(log_files, show_validation=True)
