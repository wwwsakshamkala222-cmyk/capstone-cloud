#!/usr/bin/env python3
"""Run just N1 twin analysis from phase3_novelty_v2."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_novelty_v2 import n1_twin_analysis

result = n1_twin_analysis()
print(f"\nOverall twin rate: {result['overall_twin_pct']:.2f}%")
if result.get('twin_inflation_macro_f1') is not None:
    print(f"Twin inflation on macro-F1: {result['twin_inflation_macro_f1']:.4f}")
if result.get('twin_inflation_accuracy') is not None:
    print(f"Twin inflation on accuracy: {result['twin_inflation_accuracy']:.6f}")
