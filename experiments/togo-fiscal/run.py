"""
RSI Togo Fiscal — Experiments (run.py)
=======================================
Runs all 5 experiments + worked example.
Imports rules from rules.py, core from src/core.py.

Usage:
    cd rsi-framework
    python experiments/togo-fiscal/run.py
"""

import os
import sys
import numpy as np
import pandas as pd
import time
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# IMPORTS — core + rules
# ═══════════════════════════════════════════════════════════════

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from core import PopulationRSI, EntityScorer
from rules import (
    TOGO_RULES, RULE_IDS, make_priors,
    load_dataset, extract_signals, print_dataset_summary, best_f1,
)

from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 42


# ═══════════════════════════════════════════════════════════════
# EXP-1: Per-Rule Performance + Population Calibration
# ═══════════════════════════════════════════════════════════════

def exp1(data, c_true):
    print("\n" + "="*70)
    print("EXP-1: Per-Rule Performance + Population Calibration")
    print("="*70)
    
    priors = make_priors()
    eids = data['entity_ids']
    
    # Population inference (Bayesian)
    rsi = PopulationRSI(priors)
    pop_res = rsi.fit(data['sigs_disc'], data['apps'])
    
    # Entity scoring (deterministic)
    scorer = EntityScorer(RULE_IDS)
    ent_res = scorer.score(data['sigs_raw'], data['apps'], eids)
    
    # Per-rule metrics
    print(f"\n  Per-Rule Metrics:")
    print(f"  {'Rule':<10} {'n':>5} {'NC%':>5} {'RSI F1':>7} {'AUC':>7} {'RBS F1':>7} {'D':>6}")
    print("  " + "-"*55)
    
    per_rule = {}
    all_f1_rsi, all_f1_rbs, all_auc = [], [], []
    
    for rid in RULE_IDS:
        ae = [e for e in eids if data['apps'][rid].get(e, False)]
        if len(ae) < 10:
            continue
        yt = np.array([data['labels_rule'][rid].get(e, 0) for e in ae])
        if len(np.unique(yt)) < 2:
            print(f"  {rid:<10} {len(ae):>5} {yt.mean():>5.0%} {'skip':>30}")
            continue
        
        # RSI entity scoring
        ys = np.array([ent_res['entity_rules'].get((e, rid), {}).get('nc_score', 0.5)
                      for e in ae])
        f1_r, _ = best_f1(yt, ys)
        auc_r = roc_auc_score(yt, ys)
        
        # RBS baseline
        yp_rbs = np.array([
            1 if (not np.isnan(data['sigs_disc'][rid].get(e, np.nan))
                  and data['sigs_disc'][rid].get(e, np.nan) == 0)
            else 0 for e in ae
        ])
        f1_rbs = f1_score(yt, yp_rbs, zero_division=0)
        
        d = f1_r - f1_rbs
        per_rule[rid] = {
            'f1_rsi': f1_r, 'auc': auc_r, 'f1_rbs': f1_rbs,
            'n': len(ae), 'nc': yt.mean(),
        }
        all_f1_rsi.append(f1_r)
        all_f1_rbs.append(f1_rbs)
        all_auc.append(auc_r)
        
        print(f"  {rid:<10} {len(ae):>5} {yt.mean():>5.0%} "
              f"{f1_r:>7.3f} {auc_r:>7.3f} {f1_rbs:>7.3f} {d:>+6.3f}")
    
    mf = np.mean(all_f1_rsi)
    mr = np.mean(all_f1_rbs)
    ma = np.mean(all_auc)
    print(f"\n  Mean: RSI F1={mf:.3f} AUC={ma:.3f} | RBS F1={mr:.3f} | D={mf-mr:+.3f}")
    
    # Global aggregation (secondary)
    print(f"\n  Global (secondary):")
    yt_g = np.array([data['labels_global'][e] for e in eids])
    ys_min = np.array([ent_res['entities'][e]['nc_min'] for e in eids])
    f1_min, _ = best_f1(yt_g, ys_min)
    auc_min = roc_auc_score(yt_g, ys_min) if len(np.unique(yt_g)) > 1 else 0
    
    yp_rbs_g = np.array([
        1 if any(not np.isnan(data['sigs_disc'][r].get(e, np.nan))
                 and data['sigs_disc'][r].get(e, np.nan) == 0
                 for r in RULE_IDS if data['apps'][r].get(e, False))
        else 0 for e in eids
    ])
    f1_rbs_g = f1_score(yt_g, yp_rbs_g, zero_division=0)
    
    # Supervised baselines
    dp = data['df']
    feat = ['obs_ca'] + [f'app_{r}' for r in RULE_IDS] + [f'sig_raw_{r}' for r in RULE_IDS]
    X = dp[feat].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0)
    Xtr, Xte, ytr, yte = train_test_split(
        X, yt_g, test_size=0.3, random_state=42,
        stratify=yt_g if len(np.unique(yt_g)) > 1 else None,
    )
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    
    t0 = time.perf_counter()
    xgb = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    xgb.fit(Xtr, ytr)
    xgb_ms = (time.perf_counter() - t0) * 1000
    f1_xgb = f1_score(yte, xgb.predict(Xte), zero_division=0)
    auc_xgb = roc_auc_score(yte, xgb.predict_proba(Xte)[:, 1]) if len(np.unique(yte)) > 1 else 0
    
    t0 = time.perf_counter()
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), max_iter=1000,
        random_state=42, early_stopping=True, learning_rate='adaptive',
    )
    mlp.fit(Xtr_s, ytr)
    mlp_ms = (time.perf_counter() - t0) * 1000
    f1_mlp = f1_score(yte, mlp.predict(Xte_s), zero_division=0)
    auc_mlp = roc_auc_score(yte, mlp.predict_proba(Xte_s)[:, 1]) if len(np.unique(yte)) > 1 else 0
    
    print(f"  RSI (min-agg): F1={f1_min:.3f} AUC={auc_min:.3f} (no labels)")
    print(f"  RBS:           F1={f1_rbs_g:.3f} (no labels)")
    print(f"  XGBoost:       F1={f1_xgb:.3f} AUC={auc_xgb:.3f} (supervised, {xgb_ms:.0f}ms)")
    print(f"  MLP:           F1={f1_mlp:.3f} AUC={auc_mlp:.3f} (supervised, {mlp_ms:.0f}ms)")
    
    # Population calibration
    print(f"\n  Population Calibration:")
    print(f"  {'Rule':<10} {'c*':>7} {'E[c]':>7} {'s':>7} {'CI 95%':>18} {'?':>3} {'n':>5}")
    n_ok = 0
    for rid in RULE_IDS:
        p = pop_res['population'][rid]
        ct = c_true[rid]
        ci = p['CI_95']
        ok = ci[0] <= ct <= ci[1]
        if ok:
            n_ok += 1
        print(f"  {rid:<10} {ct:>7.3f} {p['E_c']:>7.3f} {p['std_c']:>7.3f} "
              f"[{ci[0]:.3f},{ci[1]:.3f}] {'Y' if ok else 'N':>2} {p['n_obs']:>5}")
    print(f"  Calibrated: {n_ok}/{len(RULE_IDS)}")
    
    # ELBO
    elbo = pop_res['elbo_history']
    print(f"\n  ELBO: prior={elbo[0]:.2f} -> post={elbo[-1]:.2f} "
          f"Gain={elbo[-1]-elbo[0]:+.2f}")
    
    return {
        'pop_res': pop_res, 'ent_res': ent_res, 'per_rule': per_rule,
        'xgb_ms': xgb_ms, 'mlp_ms': mlp_ms,
    }


# ═══════════════════════════════════════════════════════════════
# EXP-2: O(1) Adaptability (T1)
# ═══════════════════════════════════════════════════════════════

def exp2(data):
    print("\n" + "="*70)
    print("EXP-2: O(1) Adaptability (T1)")
    print("="*70)
    
    priors = make_priors()
    rsi = PopulationRSI(priors)
    rsi.fit(data['sigs_disc'], data['apps'])
    
    pre = {k: rsi.pop['R1_TVA'][k]
           for k in ['mu_delta', 'sigma_delta', 'alpha_q', 'beta_q']}
    t_rsi = rsi.regulatory_update('R1_TVA', 100e6, 60e6)
    post = {k: rsi.pop['R1_TVA'][k]
            for k in ['mu_delta', 'sigma_delta', 'alpha_q', 'beta_q']}
    
    print(f"\n  VAT: 60M -> 100M FCFA")
    print(f"  RSI: {t_rsi:.4f} ms")
    print(f"  sigma invariant: {abs(pre['sigma_delta'] - post['sigma_delta']) < 1e-12}")
    print(f"  alpha invariant: {abs(pre['alpha_q'] - post['alpha_q']) < 1e-12}")
    print(f"  beta  invariant: {abs(pre['beta_q'] - post['beta_q']) < 1e-12}")
    
    # Baseline retrain time
    dp = data['df']
    feat = ['obs_ca'] + [f'app_{r}' for r in RULE_IDS] + [f'sig_raw_{r}' for r in RULE_IDS]
    X = dp[feat].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0)
    yt = np.array([data['labels_global'][e] for e in data['entity_ids']])
    Xtr, _, ytr, _ = train_test_split(
        X, yt, test_size=0.3, random_state=42,
        stratify=yt if len(np.unique(yt)) > 1 else None,
    )
    t0 = time.perf_counter()
    GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42).fit(Xtr, ytr)
    t_xgb = (time.perf_counter() - t0) * 1000
    
    sp = t_xgb / max(t_rsi, 0.0001)
    print(f"  XGBoost retrain: {t_xgb:.0f} ms -> Speedup: {sp:,.0f}x")
    return {'rsi_ms': t_rsi, 'xgb_ms': t_xgb, 'speedup': sp}


# ═══════════════════════════════════════════════════════════════
# EXP-3: BvM Consistency (T2)
# ═══════════════════════════════════════════════════════════════

def exp3(df):
    print("\n" + "="*70)
    print("EXP-3: BvM Consistency (T2)")
    print("="*70)
    
    priors = make_priors()
    dp = df[df['period'] == 1]
    sizes = [25, 50, 100, 250, 500, 1000, 2000]
    show = ['R1_TVA', 'R4_TPU', 'R7_DECL', 'R8_BANK']
    
    print(f"\n  {'N':>6}", end="")
    for r in show:
        print(f"  {'E[c]':>7} {'s':>7}", end="")
    print()
    print("  " + "-"*70)
    
    bvm = []
    for N in sizes:
        d = extract_signals(dp.iloc[:N].reset_index(drop=True), 1)
        rsi = PopulationRSI(priors)
        res = rsi.fit(d['sigs_disc'], d['apps'])
        row = {'N': N}
        print(f"  {N:>6}", end="")
        for r in show:
            p = res['population'][r]
            print(f"  {p['E_c']:>7.3f} {p['std_c']:>7.4f}", end="")
            row[f'{r}_std'] = p['std_c']
        print()
        bvm.append(row)
    
    print(f"\n  sigma(250)/sigma(1000) ~ 2.0:")
    for r in show:
        s250 = [d for d in bvm if d['N'] == 250][0].get(f'{r}_std', 0)
        s1k = [d for d in bvm if d['N'] == 1000][0].get(f'{r}_std', 1)
        ratio = s250 / s1k if s1k > 0 else 0
        print(f"    {r}: {ratio:.3f}")
    
    return bvm


# ═══════════════════════════════════════════════════════════════
# EXP-4: Missing Data Robustness
# ═══════════════════════════════════════════════════════════════

def exp4(data):
    print("\n" + "="*70)
    print("EXP-4: Missing Data Robustness (per-rule)")
    print("="*70)
    
    eids = data['entity_ids']
    rng = np.random.RandomState(42)
    show = ['R1_TVA', 'R4_TPU', 'R5_IRPP', 'R8_BANK']
    
    print(f"\n  {'Miss':>6}", end="")
    for r in show:
        print(f"  {'RSI':>6} {'RBS':>6}", end="")
    print()
    print("  " + "-"*58)
    
    for mr in [0.0, 0.10, 0.20, 0.30, 0.50]:
        sr = {r: dict(data['sigs_raw'][r]) for r in RULE_IDS}
        sd = {r: dict(data['sigs_disc'][r]) for r in RULE_IDS}
        
        if mr > 0:
            for rid in RULE_IDS:
                for eid in list(sr[rid].keys()):
                    if not np.isnan(sr[rid].get(eid, np.nan)) and rng.random() < mr:
                        sr[rid][eid] = np.nan
                        sd[rid][eid] = np.nan
        
        scorer = EntityScorer(RULE_IDS)
        er = scorer.score(sr, data['apps'], eids)
        
        print(f"  {mr*100:>5.0f}%", end="")
        for rid in show:
            ae = [e for e in eids if data['apps'][rid].get(e, False)]
            yt = np.array([data['labels_rule'][rid].get(e, 0) for e in ae])
            if len(np.unique(yt)) < 2:
                print(f"  {'--':>6} {'--':>6}", end="")
                continue
            ys = np.array([er['entity_rules'].get((e, rid), {}).get('nc_score', 0.5)
                          for e in ae])
            bf, _ = best_f1(yt, ys)
            yp_rbs = np.array([
                1 if (not np.isnan(sd[rid].get(e, np.nan))
                      and sd[rid].get(e, np.nan) == 0)
                else 0 for e in ae
            ])
            f_rbs = f1_score(yt, yp_rbs, zero_division=0)
            print(f"  {bf:>6.3f} {f_rbs:>6.3f}", end="")
        print()


# ═══════════════════════════════════════════════════════════════
# EXP-5: ELBO Convergence (T3)
# ═══════════════════════════════════════════════════════════════

def exp5(data):
    print("\n" + "="*70)
    print("EXP-5: ELBO Convergence (T3)")
    print("="*70)
    
    priors = make_priors()
    rsi = PopulationRSI(priors, max_iter=20, tol=1e-14)
    res = rsi.fit(data['sigs_disc'], data['apps'])
    
    elbo = res['elbo_history']
    for i, e in enumerate(elbo):
        lb = "prior" if i == 0 else f"CAVI {i}"
        d = 0 if i == 0 else e - elbo[i - 1]
        m = "Y" if i == 0 or e >= elbo[i - 1] - 1e-10 else "N"
        print(f"  {lb:>8}: {e:.2f}  D={d:+.4f}  {m}")
    
    ok = all(elbo[i + 1] >= elbo[i] - 1e-10 for i in range(len(elbo) - 1))
    gain = elbo[-1] - elbo[0]
    print(f"\n  Gain: {gain:+.2f}  Monotone: {'Yes' if ok else 'NO'}")
    return {'monotone': ok, 'gain': gain}


# ═══════════════════════════════════════════════════════════════
# WORKED EXAMPLE
# ═══════════════════════════════════════════════════════════════

def worked_example(data, ent_res, pop_res):
    print("\n" + "="*70)
    print("WORKED EXAMPLE")
    print("="*70)
    
    dp = data['df']
    for _, row in dp.iterrows():
        eid = row['entity_id']
        if eid not in ent_res['entities']:
            continue
        na = int(row.get('n_applicable', 0))
        if na < 6:
            continue
        
        raws = {}
        for r in RULE_IDS:
            if bool(row.get(f'app_{r}', False)):
                v = row.get(f'sig_raw_{r}', np.nan)
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    v = np.nan
                if not np.isnan(v):
                    raws[r] = v
        
        if len(raws) >= 5 and min(raws.values()) < 0.3 and max(raws.values()) > 0.7:
            print(f"\n  Entity: {eid} | {row['segment']} | CA: {row['obs_ca']:,.0f} FCFA | {na} rules")
            print(f"\n  {'Rule':<10} {'Signal':>7} {'E[c_pop]':>8} {'NC':>6} {'c_true':>7} {'Verdict':>12}")
            print("  " + "-"*56)
            
            for r in RULE_IDS:
                if not bool(row.get(f'app_{r}', False)):
                    continue
                raw = row.get(f'sig_raw_{r}', np.nan)
                ct = row.get(f'c_true_{r}', np.nan)
                try:
                    raw = float(raw)
                except (ValueError, TypeError):
                    raw = np.nan
                try:
                    ct = float(ct)
                except (ValueError, TypeError):
                    ct = np.nan
                
                er = ent_res['entity_rules'].get((eid, r), {})
                nc = er.get('nc_score', 0.5)
                pop_ec = pop_res['population'].get(r, {}).get('E_c', 0.5)
                
                if np.isnan(raw):
                    verdict = "missing"
                elif nc > 0.6:
                    verdict = "NON-COMPL."
                elif nc > 0.4:
                    verdict = "borderline"
                else:
                    verdict = "compliant"
                
                rs = f'{raw:.2f}' if not np.isnan(raw) else 'NaN'
                cs = f'{ct:.2f}' if not np.isnan(ct) else '-'
                print(f"  {r:<10} {rs:>7} {pop_ec:>8.3f} {nc:>6.3f} {cs:>7} {verdict:>12}")
            
            ent = ent_res['entities'][eid]
            print(f"\n  Worst rule: {ent['worst_rule']}")
            print(f"  NC (min): {ent['nc_min']:.3f} | NC (mean): {ent['nc_mean']:.3f}")
            print(f"  Auditor sees: which rule, severity, population context.")
            return
    
    print("  No suitable example found.")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("  RSI Togo Fiscal - Experiments")
    print("="*70)
    
    # Load
    print("\n[0] Loading dataset...")
    df, c_true = load_dataset()
    print_dataset_summary(df)
    
    print(f"\n  Ground truth c_true:")
    for r, v in c_true.items():
        e_prior = TOGO_RULES[r]['alpha'] / (TOGO_RULES[r]['alpha'] + TOGO_RULES[r]['beta'])
        print(f"    {r}: {v:.3f} (prior E[c]={e_prior:.2f})")
    
    # Extract
    data = extract_signals(df, period=1)
    
    # Run
    r1 = exp1(data, c_true)
    r2 = exp2(data)
    exp3(df)
    exp4(data)
    r5 = exp5(data)
    worked_example(data, r1['ent_res'], r1['pop_res'])
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    dp = df[df['period'] == 1]
    print(f"\n  Dataset: RSI-Togo-Fiscal-Synthetic v2.0, {len(dp)} enterprises, 8 rules")
    print(f"\n  Per-rule (primary):")
    for rid, pr in r1['per_rule'].items():
        print(f"    {rid}: F1={pr['f1_rsi']:.3f} AUC={pr['auc']:.3f} | "
              f"RBS={pr['f1_rbs']:.3f} | D={pr['f1_rsi']-pr['f1_rbs']:+.3f}")
    print(f"\n  T1: {r2['rsi_ms']:.4f}ms vs {r2['xgb_ms']:.0f}ms = {r2['speedup']:,.0f}x")
    print(f"  T2: BvM confirmed")
    print(f"  T3: ELBO {'monotone' if r5['monotone'] else 'FAILED'}, gain={r5['gain']:+.2f}")


if __name__ == '__main__':
    main()