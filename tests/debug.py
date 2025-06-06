#!/usr/bin/env python3
"""
Standalone script to debug TFT model parameters.
"""
import sys
import os
sys.path.insert(0, 'src')

import torch
from models import get_model
from config.model_configs import get_config

# Import your debugging functions (copy them into this file or import)
from debug_params import *  # All the debugging functions I provided

def main():
    # Use same config as your training
    config = get_config('small')  # or whatever preset you're using
    
    # Create your model
    model = get_model('tft-dict', config)  # or 'tft'
    
    print("🐛 DEBUGGING TFT MODEL PARAMETERS")
    print("=" * 60)
    
    # Run debugging
    quick_param_check(model)
    full_parameter_debug(model, config)
    
    # Compare with vanilla if needed
    print("\n🆚 COMPARING WITH VANILLA")
    vanilla_model = get_model('vanilla', config)
    compare_tft_vs_vanilla(model, vanilla_model)

if __name__ == "__main__":
    main()