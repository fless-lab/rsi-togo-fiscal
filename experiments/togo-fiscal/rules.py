"""
RSI Togo Fiscal — Domain Adapter (rules.py)
============================================
Contains ONLY Togolese fiscal domain knowledge:
- 8 rules with priors, thresholds, applicability conditions
- Signal extraction from CSV
- No experiments, no metrics, no baselines

Usage:
    from rules import TOGO_RULES, RULE_IDS, make_priors, extract_signals
"""

import os
import sys
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# PATH SETUP — find core.py in src/
# ═══════════════════════════════════════════════════════════════

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

# Default dataset path
DEFAULT_DATASET = os.path.join(ROOT_DIR, 'datasets', 'togo-fiscal', 'dataset-v2.csv')

# ═══════════════════════════════════════════════════════════════
# TOGO FISCAL RULES — institutional knowledge from OTR
# ═══════════════════════════════════════════════════════════════

TOGO_RULES = {
    'R1_TVA': {
        'alpha': 7, 'beta': 3, 'sigma_drift': 2.0,
        'thr_p1': 60e6, 'thr_p2': 100e6, 'cond': 'above',
        'desc': 'Value Added Tax (TVA), 18%, threshold 60M->100M FCFA',
    },
    'R2_IS': {
        'alpha': 6, 'beta': 4, 'sigma_drift': 1.5,
        'thr_p1': 100e6, 'thr_p2': 100e6, 'cond': 'above',
        'desc': 'Corporate Income Tax (IS), 27%, threshold 100M FCFA',
    },
    'R3_IMF': {
        'alpha': 7, 'beta': 3, 'sigma_drift': 1.5,
        'thr_p1': 60e6, 'thr_p2': 60e6, 'cond': 'above',
        'desc': 'Minimum Tax (IMF), 1%, threshold 60M FCFA',
    },
    'R4_TPU': {
        'alpha': 5, 'beta': 5, 'sigma_drift': 0.5,
        'thr_p1': 60e6, 'thr_p2': 60e6, 'cond': 'below',
        'desc': 'Informal Sector Tax (TPU), flat amount, CA < 60M',
    },
    'R5_IRPP': {
        'alpha': 6, 'beta': 4, 'sigma_drift': 1.0,
        'thr_p1': 30e6, 'thr_p2': 30e6, 'cond': 'above',
        'desc': 'Income Tax Withholding (IRPP), 15%, threshold 30M',
    },
    'R6_PAT': {
        'alpha': 7, 'beta': 3, 'sigma_drift': 0.5,
        'thr_p1': 30e6, 'thr_p2': 30e6, 'cond': 'above',
        'desc': 'Business License (Patente), threshold 30M',
    },
    'R7_DECL': {
        'alpha': 6, 'beta': 4, 'sigma_drift': 0.5,
        'thr_p1': 0, 'thr_p2': 0, 'cond': 'all',
        'desc': 'Annual Declaration Obligation (universal)',
    },
    'R8_BANK': {
        'alpha': 5, 'beta': 5, 'sigma_drift': 0.5,
        'thr_p1': 0, 'thr_p2': 0, 'cond': 'all',
        'desc': 'Bank Formalization (universal)',
    },
}

RULE_IDS = list(TOGO_RULES.keys())


# ═══════════════════════════════════════════════════════════════
# PRIORS — convert domain rules to core format
# ═══════════════════════════════════════════════════════════════

def make_priors():
    """Convert TOGO_RULES to the format expected by PopulationRSI."""
    return {
        rid: {
            'alpha': cfg['alpha'],
            'beta': cfg['beta'],
            'sigma_drift': cfg['sigma_drift'],
        }
        for rid, cfg in TOGO_RULES.items()
    }


# ═══════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════

def load_dataset(path=None):
    """
    Load the RSI-Togo-Fiscal-Synthetic v2.0 CSV.
    
    Returns
    -------
    df : DataFrame
    c_true : dict {rule_id: estimated population compliance rate}
    """
    path = path or DEFAULT_DATASET
    if not os.path.exists(path):
        print(f"ERROR: Dataset not found at {path}")
        print(f"  Run: python datasets/togo-fiscal/script/generate_dataset.py")
        sys.exit(1)
    
    df = pd.read_csv(path)
    
    # Estimate c_true from entity-level ground truth
    dp = df[df['period'] == 1]
    c_true = {}
    for rid in RULE_IDS:
        vals = dp[dp[f'app_{rid}'] == True][f'c_true_{rid}'].dropna()
        if len(vals) > 0:
            c_true[rid] = vals.mean()
        else:
            cfg = TOGO_RULES[rid]
            c_true[rid] = cfg['alpha'] / (cfg['alpha'] + cfg['beta'])
    
    return df, c_true


# ═══════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION — from CSV to core-compatible format
# ═══════════════════════════════════════════════════════════════

def extract_signals(df, period=1):
    """
    Extract signals, applicability, and labels from the dataset.
    
    Parameters
    ----------
    df : DataFrame (full dataset)
    period : int (1 or 2)
    
    Returns
    -------
    dict with keys:
        sigs_disc  : {rule_id: {entity_id: 0.0 or 1.0 or NaN}}
        sigs_raw   : {rule_id: {entity_id: float in [0,1] or NaN}}
        apps       : {rule_id: {entity_id: bool}}
        labels_global : {entity_id: int}  (1 if any rule NC)
        labels_rule   : {rule_id: {entity_id: int}}  (1 if this rule NC)
        ct_rule       : {(entity_id, rule_id): float}  (ground truth c_true)
        df         : DataFrame (filtered to this period)
        entity_ids : sorted list of entity IDs
    """
    dp = df[df['period'] == period].copy()
    
    sigs_disc = {r: {} for r in RULE_IDS}
    sigs_raw = {r: {} for r in RULE_IDS}
    apps = {r: {} for r in RULE_IDS}
    labels_global = {}
    labels_rule = {r: {} for r in RULE_IDS}
    ct_rule = {}
    
    for _, row in dp.iterrows():
        eid = row['entity_id']
        labels_global[eid] = int(row.get('label_any_nc', 0))
        
        for rid in RULE_IDS:
            app = bool(row.get(f'app_{rid}', False))
            apps[rid][eid] = app
            
            if app:
                # Discretized signal
                sd = row.get(f'sig_disc_{rid}', np.nan)
                try:
                    sd = float(sd)
                except (ValueError, TypeError):
                    sd = np.nan
                sigs_disc[rid][eid] = sd
                
                # Raw continuous signal
                sr = row.get(f'sig_raw_{rid}', np.nan)
                try:
                    sr = float(sr)
                except (ValueError, TypeError):
                    sr = np.nan
                sigs_raw[rid][eid] = sr
                
                # Per-rule label
                lbl = int(row.get(f'label_{rid}', 0))
                labels_rule[rid][eid] = lbl
                
                # Ground truth compliance
                ct = row.get(f'c_true_{rid}', np.nan)
                try:
                    ct = float(ct)
                except (ValueError, TypeError):
                    ct = np.nan
                ct_rule[(eid, rid)] = ct
    
    entity_ids = sorted(labels_global.keys())
    
    return {
        'sigs_disc': sigs_disc,
        'sigs_raw': sigs_raw,
        'apps': apps,
        'labels_global': labels_global,
        'labels_rule': labels_rule,
        'ct_rule': ct_rule,
        'df': dp,
        'entity_ids': entity_ids,
    }


# ═══════════════════════════════════════════════════════════════
# DATASET SUMMARY — for quick inspection
# ═══════════════════════════════════════════════════════════════

def print_dataset_summary(df):
    """Print a summary of the dataset structure."""
    dp = df[df['period'] == 1]
    n = len(dp)
    
    print(f"  Enterprises: {n}")
    print(f"  Periods: {sorted(df['period'].unique())}")
    print(f"  Segments: {dp['segment'].value_counts().to_dict()}")
    
    print(f"\n  Rules:")
    for r in RULE_IDS:
        na = dp[f'app_{r}'].sum()
        nc = dp[dp[f'app_{r}'] == True][f'label_{r}'].sum()
        nm = dp[dp[f'app_{r}'] == True][f'sig_disc_{r}'].isna().sum()
        desc = TOGO_RULES[r]['desc']
        print(f"    {r}: {na:>5} app ({na/n:>3.0%}), "
              f"{nc:>4} NC ({nc/max(na,1):>3.0%}), "
              f"{nm:>4} miss ({nm/max(na,1):>3.0%})  [{desc}]")
    
    print(f"\n  Rules per entity: "
          f"{dp['n_applicable'].value_counts().sort_index().to_dict()}")


def best_f1(yt, ys):
    """Find optimal F1 threshold (utility used by run.py)."""
    from sklearn.metrics import f1_score as sk_f1
    bf, bt = 0, 0.5
    for tau in np.linspace(0.05, 0.95, 181):
        f = sk_f1(yt, (ys >= tau).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, tau
    return bf, bt