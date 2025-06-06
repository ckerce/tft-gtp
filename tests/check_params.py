import sys
sys.path.insert(0, 'src')
from debug_params import analyze_pt_file_parameters

# Replace with your actual .pt file path
pt_file = "./outputs/training_run/tft-dict_model.pt"  # Your model file
analyze_pt_file_parameters(pt_file)