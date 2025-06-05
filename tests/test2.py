#!/usr/bin/env python3
"""
Compare initial (untrained) vs final (trained) dictionary loss to see training progression.
Infers LM loss from total - dict_loss and renames dict to ffn for clarity.
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
    print(f"🔤 FFN vocab size: {getattr(config, 'dict_vocab_size', 'Unknown')}")
    print(f"⚖️ FFN loss weight: {getattr(config, 'dict_loss_weight', 'Unknown')}")
    
    # Create tokenizer
    tokenizer = create_tokenizer('gpt2')
    
    # Test texts - using simple texts as in your script
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
    
    initial_totals = {'total': 0, 'lm': 0, 'ffn': 0, 'count': 0}
    
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            input_ids = torch.tensor([tokenizer.encode(text)])
            outputs = initial_model(input_ids, labels=input_ids)
            
            total_loss = outputs.get('loss')
            dict_loss = outputs.get('dict_loss')  # Still called dict_loss in model output
            
            # Debug output
            print(f"    Debug - Raw outputs: loss={total_loss}, dict_loss={dict_loss}")
            
            if total_loss is not None and dict_loss is not None:
                total_val = total_loss.item()
                ffn_val = dict_loss.item() if hasattr(dict_loss, 'item') else dict_loss
                
                # INFER LM LOSS: LM = Total - (FFN_weight * FFN_loss)
                weighted_ffn = ffn_val * config.dict_loss_weight
                lm_val = total_val - weighted_ffn
                
                ratio = ffn_val/lm_val if lm_val != 0 else float('inf')
                print(f"  Text {i+1}: Total={total_val:.4f}, LM={lm_val:.4f}, FFN={ffn_val:.4f}, Ratio={ratio:.4f}")
                print(f"           Weighted FFN={weighted_ffn:.4f}, Check: LM+wFFN={lm_val+weighted_ffn:.4f}")
                
                initial_totals['total'] += total_val
                initial_totals['lm'] += lm_val
                initial_totals['ffn'] += ffn_val
                initial_totals['count'] += 1
    
    # === FINAL MODEL (Trained) ===
    print(f"\n" + "="*60)
    print(f"🎯 FINAL MODEL (Trained)")
    print(f"="*60)
    
    final_model = get_model('tft-dict', config)
    final_model.load_state_dict(checkpoint['model_state_dict'])
    final_model.eval()
    
    final_totals = {'total': 0, 'lm': 0, 'ffn': 0, 'count': 0}
    
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            input_ids = torch.tensor([tokenizer.encode(text)])
            outputs = final_model(input_ids, labels=input_ids)
            
            total_loss = outputs.get('loss')
            dict_loss = outputs.get('dict_loss')
            
            # Debug output
            print(f"    Debug - Raw outputs: loss={total_loss}, dict_loss={dict_loss}")
            
            if total_loss is not None and dict_loss is not None:
                total_val = total_loss.item()
                ffn_val = dict_loss.item() if hasattr(dict_loss, 'item') else dict_loss
                
                # INFER LM LOSS: LM = Total - (FFN_weight * FFN_loss)
                weighted_ffn = ffn_val * config.dict_loss_weight
                lm_val = total_val - weighted_ffn
                
                ratio = ffn_val/lm_val if lm_val != 0 else float('inf')
                print(f"  Text {i+1}: Total={total_val:.4f}, LM={lm_val:.4f}, FFN={ffn_val:.4f}, Ratio={ratio:.4f}")
                print(f"           Weighted FFN={weighted_ffn:.4f}, Check: LM+wFFN={lm_val+weighted_ffn:.4f}")
                
                final_totals['total'] += total_val
                final_totals['lm'] += lm_val
                final_totals['ffn'] += ffn_val
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
        print(f"  FFN Loss:        {init_avg['ffn']:7.4f}  {final_avg['ffn']:7.4f}  {init_avg['ffn']-final_avg['ffn']:+7.4f} ({((init_avg['ffn']-final_avg['ffn'])/init_avg['ffn']*100) if init_avg['ffn'] != 0 else 0:+5.1f}%)")
        
        print(f"\n🔍 LOSS COMPOSITION:")
        print(f"                    Initial             Final")
        init_weighted_ffn = init_avg['ffn'] * config.dict_loss_weight
        final_weighted_ffn = final_avg['ffn'] * config.dict_loss_weight
        print(f"  LM Loss:         {init_avg['lm']:7.4f} ({init_avg['lm']/init_avg['total']*100:5.1f}%)   {final_avg['lm']:7.4f} ({final_avg['lm']/final_avg['total']*100:5.1f}%)")
        print(f"  Weighted FFN:    {init_weighted_ffn:7.4f} ({init_weighted_ffn/init_avg['total']*100:5.1f}%)   {final_weighted_ffn:7.4f} ({final_weighted_ffn/final_avg['total']*100:5.1f}%)")
        print(f"  FFN Loss Weight: {config.dict_loss_weight}")
        
        print(f"\n🔍 RATIOS:")
        init_ratio = init_avg['ffn'] / init_avg['lm'] if init_avg['lm'] > 0 else 0
        final_ratio = final_avg['ffn'] / final_avg['lm'] if final_avg['lm'] > 0 else 0
        print(f"  FFN/LM Initial:  {init_ratio:.4f}")
        print(f"  FFN/LM Final:    {final_ratio:.4f}")
        print(f"  Ratio Change:    {final_ratio - init_ratio:+.4f}")
        
        print(f"\n🎯 ANALYSIS:")
        
        # FFN constraint analysis
        if final_avg['ffn'] < 0.01:
            print(f"  ✅ Excellent final FFN constraint (ffn_loss < 0.01)")
            print(f"     → FFN outputs perfectly align with vocabulary")
        elif final_avg['ffn'] < 0.1:
            print(f"  ✅ Good final FFN constraint (ffn_loss < 0.1)")
            print(f"     → FFN outputs well-aligned with vocabulary")
        elif final_avg['ffn'] < 0.5:
            print(f"  ⚠️ Moderate final FFN constraint")
            print(f"     → Some misalignment between FFN and vocabulary")
        else:
            print(f"  ❌ Poor final FFN constraint")
            print(f"     → FFN outputs poorly aligned with vocabulary")
        
        # Training dynamics analysis
        ffn_improvement = ((init_avg['ffn'] - final_avg['ffn']) / init_avg['ffn']) * 100 if init_avg['ffn'] != 0 else 0
        lm_improvement = ((init_avg['lm'] - final_avg['lm']) / init_avg['lm']) * 100 if init_avg['lm'] != 0 else 0
        
        print(f"\n📈 TRAINING DYNAMICS:")
        print(f"  LM improvement:  {lm_improvement:+5.1f}%")
        print(f"  FFN improvement: {ffn_improvement:+5.1f}%")
        
        if ffn_improvement > lm_improvement:
            print(f"  📚 FFN constraint learned faster than language modeling")
            print(f"     → Model quickly learned vocabulary-aligned representations")
        elif lm_improvement > ffn_improvement:
            print(f"  🎯 Language modeling improved faster than FFN constraint")
            print(f"     → Model prioritized prediction accuracy over vocabulary alignment")
        else:
            print(f"  ⚖️ Balanced improvement in both components")
        
        # Overall training success
        if final_avg['total'] < init_avg['total'] * 0.5:
            print(f"  🚀 Excellent training progress (>50% total loss reduction)")
        elif final_avg['total'] < init_avg['total'] * 0.8:
            print(f"  ✅ Good training progress (>20% total loss reduction)")
        else:
            print(f"  ⚠️ Limited training progress (<20% total loss reduction)")
        
        # FFN constraint effectiveness
        ffn_contribution = final_weighted_ffn / final_avg['total'] if final_avg['total'] != 0 else 0
        if ffn_contribution < 0.05:
            print(f"  ⚖️ FFN constraint is well-satisfied (contributes {ffn_contribution*100:.1f}% to total loss)")
            print(f"     → Dictionary reconstruction nearly perfect")
        elif ffn_contribution < 0.15:
            print(f"  ⚖️ FFN constraint is moderately active (contributes {ffn_contribution*100:.1f}% to total loss)")
            print(f"     → Good vocabulary alignment with some flexibility")
        else:
            print(f"  ⚖️ FFN constraint dominates training (contributes {ffn_contribution*100:.1f}% to total loss)")
            print(f"     → May be over-constraining the model")
        
        # Architecture insight
        print(f"\n🏗️ ARCHITECTURE INSIGHT:")
        if final_avg['ffn'] < 0.01 and final_avg['lm'] < init_avg['lm'] * 0.7:
            print(f"  🎯 OPTIMAL: Low FFN loss + good LM improvement")
            print(f"     → Dictionary constraint guides learning without hurting performance")
        elif final_avg['ffn'] < 0.1 and final_avg['lm'] < init_avg['lm'] * 0.8:
            print(f"  ✅ GOOD: Moderate FFN constraint with decent LM performance")
        else:
            print(f"  ⚠️ SUBOPTIMAL: Either poor FFN alignment or limited LM improvement")

if __name__ == "__main__":
    compare_initial_vs_final()