#!/usr/bin/env python3
"""
Compare initial (untrained) vs final (trained) dictionary loss to see training progression.
"""

import torch
import sys
sys.path.insert(0, 'src')

from models import get_model
from mytokenizers import create_tokenizer
import copy

# === MODIFY THIS PATH ===
CHECKPOINT_PATH = "./outputs/wiki_dict/tft_dict_factored/tft-dict_model.pt"  # Your path here

def compare_initial_vs_final():
    """Compare initial vs final dictionary loss."""
    
    print(f"🔍 Loading checkpoint: {CHECKPOINT_PATH}")
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    config = checkpoint['config']
    
    print(f"📊 Model: {config.n_layers}L-{config.n_heads}H-{config.d_model}D")
    print(f"🔤 Dict vocab size: {getattr(config, 'dict_vocab_size', 'Unknown')}")
    print(f"⚖️ Dict loss weight: {getattr(config, 'dict_loss_weight', 'Unknown')}")
    
    # Create tokenizer
    tokenizer = create_tokenizer('gpt2')
    
    # Test texts
    test_texts = [
        "The transformer architecture revolutionized natural language processing.",
        "Machine learning models require careful optimization and tuning.",
        "Artificial intelligence systems can process vast amounts of data.",
        "Neural networks learn complex patterns through gradient descent.",
        "Dictionary learning provides interpretable feature representations."
    ]
    test_texts = [
        "The cat is happy.",
        "I like pizza.",
        "Today is sunny.",
        "She reads books.",
        "The dog runs fast."
    ]
        
    print(f"\n🧪 Testing on {len(test_texts)} sample texts...")
    
    # === INITIAL MODEL (Untrained) ===
    print(f"\n" + "="*60)
    print(f"🌱 INITIAL MODEL (Randomly Initialized)")
    print(f"="*60)
    
    initial_model = get_model('tft-dict', config)
    initial_model.eval()
    
    initial_totals = {'total': 0, 'lm': 0, 'dict': 0, 'count': 0}
    
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            input_ids = torch.tensor([tokenizer.encode(text)])
            outputs = initial_model(input_ids, labels=input_ids)
            
            total_loss = outputs.get('loss')
            lm_loss = outputs.get('lm_loss')
            dict_loss = outputs.get('dict_loss')
            
            # Debug output
            print(f"    Debug - Raw outputs: loss={total_loss}, lm_loss={lm_loss}, dict_loss={dict_loss}")
            
            if total_loss is not None:
                total_val = total_loss.item()
                lm_val = lm_loss.item() if lm_loss is not None else 0
                dict_val = dict_loss.item() if hasattr(dict_loss, 'item') else dict_loss if dict_loss is not None else 0
                
                ratio = dict_val/lm_val if lm_val != 0 else float('inf')
                print(f"  Text {i+1}: Total={total_val:.4f}, LM={lm_val:.4f}, Dict={dict_val:.4f}, Ratio={ratio:.4f}")
                
                initial_totals['total'] += total_val
                initial_totals['lm'] += lm_val
                initial_totals['dict'] += dict_val
                initial_totals['count'] += 1
    
    # === FINAL MODEL (Trained) ===
    print(f"\n" + "="*60)
    print(f"🎯 FINAL MODEL (Trained)")
    print(f"="*60)
    
    final_model = get_model('tft-dict', config)
    final_model.load_state_dict(checkpoint['model_state_dict'])
    final_model.eval()
    
    final_totals = {'total': 0, 'lm': 0, 'dict': 0, 'count': 0}
    
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            input_ids = torch.tensor([tokenizer.encode(text)])
            outputs = final_model(input_ids, labels=input_ids)
            
            total_loss = outputs.get('loss')
            lm_loss = outputs.get('lm_loss')
            dict_loss = outputs.get('dict_loss')
            
            # Debug output
            print(f"    Debug - Raw outputs: loss={total_loss}, lm_loss={lm_loss}, dict_loss={dict_loss}")
            
            if total_loss is not None:
                total_val = total_loss.item()
                lm_val = lm_loss.item() if lm_loss is not None else 0
                dict_val = dict_loss.item() if hasattr(dict_loss, 'item') else dict_loss if dict_loss is not None else 0
                
                ratio = dict_val/lm_val if lm_val != 0 else float('inf')
                print(f"  Text {i+1}: Total={total_val:.4f}, LM={lm_val:.4f}, Dict={dict_val:.4f}, Ratio={ratio:.4f}")
                
                final_totals['total'] += total_val
                final_totals['lm'] += lm_val
                final_totals['dict'] += dict_val
                final_totals['count'] += 1
    
    # === COMPARISON SUMMARY ===
    print(f"\n" + "="*60)
    print(f"📊 TRAINING PROGRESS SUMMARY")
    print(f"="*60)
    
    if initial_totals['count'] > 0 and final_totals['count'] > 0:
        # Calculate averages
        init_avg = {k: v/initial_totals['count'] if k != 'count' else v for k, v in initial_totals.items()}
        final_avg = {k: v/final_totals['count'] if k != 'count' else v for k, v in final_totals.items()}
        
        print(f"📈 AVERAGE LOSSES:")
        print(f"                    Initial    Final     Improvement")
        print(f"  Total Loss:      {init_avg['total']:7.4f}  {final_avg['total']:7.4f}  {init_avg['total']-final_avg['total']:+7.4f} ({((init_avg['total']-final_avg['total'])/init_avg['total']*100) if init_avg['total'] != 0 else 0:+5.1f}%)")
        print(f"  LM Loss:         {init_avg['lm']:7.4f}  {final_avg['lm']:7.4f}  {init_avg['lm']-final_avg['lm']:+7.4f} ({((init_avg['lm']-final_avg['lm'])/init_avg['lm']*100) if init_avg['lm'] != 0 else 0:+5.1f}%)")
        print(f"  Dict Loss:       {init_avg['dict']:7.4f}  {final_avg['dict']:7.4f}  {init_avg['dict']-final_avg['dict']:+7.4f} ({((init_avg['dict']-final_avg['dict'])/init_avg['dict']*100) if init_avg['dict'] != 0 else 0:+5.1f}%)")
        
        print(f"\n🔍 RATIOS:")
        init_ratio = init_avg['dict'] / init_avg['lm'] if init_avg['lm'] > 0 else 0
        final_ratio = final_avg['dict'] / final_avg['lm'] if final_avg['lm'] > 0 else 0
        print(f"  Dict/LM Initial: {init_ratio:.4f}")
        print(f"  Dict/LM Final:   {final_ratio:.4f}")
        print(f"  Ratio Change:    {final_ratio - init_ratio:+.4f}")
        
        print(f"\n🎯 ANALYSIS:")
        
        # Dictionary alignment analysis
        if final_avg['dict'] < 0.01:
            print(f"  ✅ Excellent final dictionary alignment (dict_loss < 0.01)")
        elif final_avg['dict'] < 0.1:
            print(f"  ✅ Good final dictionary alignment (dict_loss < 0.1)")
        elif final_avg['dict'] < 0.5:
            print(f"  ⚠️ Moderate final dictionary alignment")
        else:
            print(f"  ❌ Poor final dictionary alignment")
        
        # Training effectiveness
        dict_improvement = ((init_avg['dict'] - final_avg['dict']) / init_avg['dict']) * 100 if init_avg['dict'] != 0 else 0
        lm_improvement = ((init_avg['lm'] - final_avg['lm']) / init_avg['lm']) * 100 if init_avg['lm'] != 0 else 0
        
        if dict_improvement > lm_improvement:
            print(f"  📚 Dictionary constraint improved faster than language modeling")
            print(f"     → Model quickly learned vocabulary-aligned representations")
        else:
            print(f"  🎯 Language modeling improved faster than dictionary constraint")
            print(f"     → Model prioritized prediction accuracy over vocabulary alignment")
        
        # Overall training success
        if final_avg['total'] < init_avg['total'] * 0.5:
            print(f"  🚀 Excellent training progress (>50% loss reduction)")
        elif final_avg['total'] < init_avg['total'] * 0.8:
            print(f"  ✅ Good training progress (>20% loss reduction)")
        else:
            print(f"  ⚠️ Limited training progress (<20% loss reduction)")
        
        # Dictionary effectiveness
        if final_avg['total'] != 0:
            effective_dict_weight = final_avg['dict'] * config.dict_loss_weight
            dict_contribution = effective_dict_weight / final_avg['total']
            if dict_contribution < 0.1:
                print(f"  ⚖️ Dictionary constraint is well-satisfied (contributes {dict_contribution*100:.1f}% to total loss)")
            elif dict_contribution < 0.3:
                print(f"  ⚖️ Dictionary constraint is moderately active (contributes {dict_contribution*100:.1f}% to total loss)")
            else:
                print(f"  ⚖️ Dictionary constraint dominates training (contributes {dict_contribution*100:.1f}% to total loss)")
        else:
            print(f"  ⚠️ Cannot calculate dictionary contribution (total loss is zero)")

if __name__ == "__main__":
    compare_initial_vs_final()