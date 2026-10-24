#!/usr/bin/env python3
"""Diagnostic: check portfolio_values in results."""
import pickle, numpy as np

with open('meta_controller_results.pkl','rb') as f:
    data = pickle.load(f)
mc = data['meta_controller']
pv = mc['portfolio_values']

print(f'=== Portfolio Values ===')
print(f'len: {len(pv)}')
print(f'first: {pv[0]:.2f}')
print(f'last: {pv[-1]:.2f}')
print(f'final_capital: {mc["final_capital"]:.2f}')
print(f'total_return: {mc["total_return"]:.4f}')

if len(pv) > 2:
    tr = pv[-1] / 100000 - 1
    nd = max(1, len(pv)-1)
    ann = (1+tr)**(252/nd) - 1
    if ann < -1: ann = -1.0
    dr = np.diff(pv) / pv[:-1]
    sharpe = np.mean(dr)/np.std(dr)*np.sqrt(252) if np.std(dr) > 0 else 0
    print(f'manual tr={tr:.4f}, nd={nd}, ann={ann:.4f}, sharpe={sharpe:.4f}')
else:
    print(f'TOO FEW VALUES (only {len(pv)}) - mark-to-market not working')
    print(f'expected: {501+1} (501 days + initial + final)')
    
print(f'\n=== Check: is pv[-1] == final_capital? ===')
print(f'pv[-1]={pv[-1]:.2f}, final_capital={mc["final_capital"]:.2f}, match={abs(pv[-1]-mc["final_capital"])<0.01}')
