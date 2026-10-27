#!/usr/bin/env python3
"""Diagnostic v2: check portfolio_values + loop structure."""
import pickle, numpy as np, ast, inspect

# Check the source code structure
with open('meta_controller_backtest.py', 'r') as f:
    source = f.read()

# Find all occurrences of "portfolio_values.append"
import re
positions = [m.start() for m in re.finditer(r'portfolio_values\.append', source)]
print(f"portfolio_values.append found at {len(positions)} locations:")
for i, pos in enumerate(positions):
    context_start = max(0, pos - 50)
    context_end = min(len(source), pos + 50)
    print(f"  #{i+1}: ...{source[context_start:context_end]}...")

# Check results
with open('meta_controller_results.pkl','rb') as f:
    data = pickle.load(f)
mc = data['meta_controller']
pv = mc['portfolio_values']
print(f'\n=== Results ===')
print(f'portfolio_values len: {len(pv)}')
print(f'first: {pv[0]:.2f}, last: {pv[-1]:.2f}')
print(f'final_capital: {mc["final_capital"]:.2f}')

if len(pv) > 3:
    tr = pv[-1] / 100000 - 1
    nd = max(1, len(pv)-1)
    ann = (1+tr)**(252/nd) - 1
    if ann < -1: ann = -1.0
    dr = np.diff(pv) / pv[:-1]
    sharpe = np.mean(dr)/np.std(dr)*np.sqrt(252) if np.std(dr) > 0 else 0
    print(f'Manual: tr={tr:.4f}, ann={ann:.4f}, sharpe={sharpe:.4f}')
    print(f'OK - portfolio_values has {len(pv)} entries')
else:
    print(f'PROBLEM: Only {len(pv)} entries (expected 502)')
