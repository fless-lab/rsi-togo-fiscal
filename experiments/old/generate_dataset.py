"""
RSI-Togo-Fiscal-Synthetic v2.0 — Dataset Generator
====================================================
Run once: python generate_dataset.py
Produces: rsi_togo_v2.csv (4000 rows = 2000 enterprises × 2 periods)

This file is NOT needed at runtime. Once the CSV exists, use rsi_togo.py.
"""

import numpy as np
import pandas as pd

TOGO_RULES = {
    'R1_TVA':  {'alpha':7,'beta':3,'thr_p1':60e6,'thr_p2':100e6,'cond':'above'},
    'R2_IS':   {'alpha':6,'beta':4,'thr_p1':100e6,'thr_p2':100e6,'cond':'above'},
    'R3_IMF':  {'alpha':7,'beta':3,'thr_p1':60e6,'thr_p2':60e6,'cond':'above'},
    'R4_TPU':  {'alpha':5,'beta':5,'thr_p1':60e6,'thr_p2':60e6,'cond':'below'},
    'R5_IRPP': {'alpha':6,'beta':4,'thr_p1':30e6,'thr_p2':30e6,'cond':'above'},
    'R6_PAT':  {'alpha':7,'beta':3,'thr_p1':30e6,'thr_p2':30e6,'cond':'above'},
    'R7_DECL': {'alpha':6,'beta':4,'thr_p1':0,'thr_p2':0,'cond':'all'},
    'R8_BANK': {'alpha':5,'beta':5,'thr_p1':0,'thr_p2':0,'cond':'all'},
}
RULE_IDS = list(TOGO_RULES.keys())

SEGMENTS = {
    'informal':     {'frac':0.45, 'log_mu':16.3, 'log_sigma':0.7},
    'small_formal': {'frac':0.25, 'log_mu':17.6, 'log_sigma':0.5},
    'medium':       {'frac':0.18, 'log_mu':18.6, 'log_sigma':0.4},
    'large':        {'frac':0.12, 'log_mu':20.0, 'log_sigma':0.5},
}


def generate(n=2000, seed=42, output='rsi_togo_v2.csv'):
    rng = np.random.RandomState(seed)
    
    # Assign segments
    segs = []
    for name, cfg in SEGMENTS.items():
        segs.extend([name] * int(n * cfg['frac']))
    while len(segs) < n:
        segs.append('informal')
    rng.shuffle(segs)
    
    # Population ground truth compliance rates
    c_true = {}
    for rid, cfg in TOGO_RULES.items():
        c_true[rid] = rng.beta(cfg['alpha'], cfg['beta'])
    
    rows = []
    for period in [1, 2]:
        for j in range(n):
            rows.append(_gen_entity(j, segs[j], period, c_true, rng))
    
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    
    # Print summary
    dp = df[df['period'] == 1]
    print(f"RSI-Togo-Fiscal-Synthetic v2.0")
    print(f"  Saved: {output}")
    print(f"  Shape: {df.shape}")
    print(f"  Enterprises: {n} × 2 periods")
    print(f"  Segments: {dp['segment'].value_counts().to_dict()}")
    print(f"\n  Ground truth c_true (seed={seed}):")
    for r, v in c_true.items():
        print(f"    {r}: {v:.4f}")
    print(f"\n  Per-rule stats (period 1):")
    for r in RULE_IDS:
        na = dp[f'app_{r}'].sum()
        nc = dp[dp[f'app_{r}'] == True][f'label_{r}'].sum()
        nm = dp[dp[f'app_{r}'] == True][f'sig_disc_{r}'].isna().sum()
        print(f"    {r}: {na:>5} applicable ({na/n:.0%}), {nc:>4} NC ({nc/max(na,1):.0%}), {nm:>4} missing ({nm/max(na,1):.0%})")
    print(f"\n  Rules per entity:")
    for k, v in dp['n_applicable'].value_counts().sort_index().items():
        print(f"    K={k}: {v} ({v/n:.0%})")
    print(f"\n  NC(any): {dp['label_any_nc'].mean():.1%}")
    
    return df, c_true


def _gen_entity(eid, segment, period, c_true, rng):
    seg = SEGMENTS[segment]
    true_ca = rng.lognormal(seg['log_mu'], seg['log_sigma'])
    beta_ratio = rng.beta(7, 3)
    obs_ca = max(500_000, true_ca * beta_ratio * rng.normal(1.0, 0.04))
    
    row = {
        'entity_id': f'E{eid:05d}',
        'period': period,
        'segment': segment,
        'true_ca': true_ca,
        'obs_ca': obs_ca,
        'under_decl_ratio': beta_ratio,
    }
    
    any_nc = False
    n_app = 0
    
    for rid, cfg in TOGO_RULES.items():
        thr = cfg['thr_p1'] if period == 1 else cfg['thr_p2']
        
        if cfg['cond'] == 'all':
            app = True
        elif cfg['cond'] == 'below':
            app = obs_ca < thr
        else:
            app = obs_ca >= thr
        
        row[f'app_{rid}'] = app
        
        if not app:
            row[f'c_true_{rid}'] = np.nan
            row[f'sig_raw_{rid}'] = np.nan
            row[f'sig_disc_{rid}'] = np.nan
            row[f'label_{rid}'] = 0
            continue
        
        n_app += 1
        
        # Entity compliance drawn from population with dispersion=5
        c_ent = rng.beta(max(0.1, 5 * c_true[rid]),
                         max(0.1, 5 * (1 - c_true[rid])))
        row[f'c_true_{rid}'] = c_ent
        
        # Unbiased signal: Beta(c_ent*κ, (1-c_ent)*κ), E[signal] = c_ent
        if rid == 'R4_TPU':
            sig = 1.0 if rng.random() < c_ent else 0.0
        else:
            kappa = 10
            sig = float(rng.beta(max(0.01, c_ent * kappa),
                                 max(0.01, (1 - c_ent) * kappa)))
        
        # Missing: 18%
        if rng.random() < 0.18:
            sig = np.nan
        
        row[f'sig_raw_{rid}'] = sig
        if sig is None or (isinstance(sig, float) and np.isnan(sig)):
            row[f'sig_disc_{rid}'] = np.nan
        else:
            row[f'sig_disc_{rid}'] = 1.0 if sig > 0.5 else 0.0
        
        nc = 1 if c_ent < 0.5 else 0
        row[f'label_{rid}'] = nc
        if nc:
            any_nc = True
    
    row['label_any_nc'] = 1 if any_nc else 0
    row['n_applicable'] = n_app
    return row


if __name__ == '__main__':
    generate()
