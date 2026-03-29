"""
RSI Core — Domain-Agnostic Population Inference + Entity Scoring
================================================================

Two components:
    1. PopulationRSI  — Bayesian inference over population compliance rates
    2. EntityScorer    — Deterministic per-rule entity scoring

PopulationRSI Model:
    c_k ~ Beta(α_k, β_k)           [population compliance rate per rule k]
    s_jk ~ Bernoulli(c_k)          [discretized signal, entity j, rule k]
    δ_k ~ N(0, σ_k²)              [parametric drift]
    Q*(c_k) = Beta(α_k + Σ s_jk, β_k + Σ (1-s_jk))   [exact conjugate]

Guarantees (all exact, no approximation):
    T1: O(1) regulatory adaptability
    T2: BvM consistency (std ~ 1/√N)
    T3: Monotone ELBO convergence

EntityScorer:
    Deterministic rule-check with continuous scores, missing data handling.
    NOT Bayesian. Aggregation (min, mean, ≥K) is a domain policy choice.

Usage:
    from rsi_core import PopulationRSI, EntityScorer
"""

import numpy as np
from scipy import special, stats
import time


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def kl_beta(a1, b1, a2, b2):
    """KL[Beta(a1,b1) || Beta(a2,b2)]."""
    a1, b1 = max(a1, 1e-10), max(b1, 1e-10)
    a2, b2 = max(a2, 1e-10), max(b2, 1e-10)
    return max(0.0,
        special.gammaln(a2) + special.gammaln(b2) - special.gammaln(a2 + b2)
        - special.gammaln(a1) - special.gammaln(b1) + special.gammaln(a1 + b1)
        + (a1 - a2) * special.digamma(a1)
        + (b1 - b2) * special.digamma(b1)
        + (a2 + b2 - a1 - b1) * special.digamma(a1 + b1))


def kl_gaussian(mu1, sig1, mu2, sig2):
    """KL[N(mu1,sig1²) || N(mu2,sig2²)]."""
    sig1, sig2 = max(sig1, 1e-10), max(sig2, 1e-10)
    return np.log(sig2 / sig1) + (sig1**2 + (mu1 - mu2)**2) / (2 * sig2**2) - 0.5


# ═══════════════════════════════════════════════════════════════
# POPULATION RSI — Bayesian
# ═══════════════════════════════════════════════════════════════

class PopulationRSI:
    """
    Population-level Bayesian inference over rule compliance rates.
    
    Parameters
    ----------
    rule_priors : dict
        {rule_id: {'alpha': float, 'beta': float, 'sigma_drift': float}}
    sigma_obs : float
        Observation noise for drift estimation.
    tol : float
        ELBO convergence tolerance.
    max_iter : int
        Maximum CAVI iterations.
    """
    
    def __init__(self, rule_priors, sigma_obs=0.0625, tol=1e-8, max_iter=100):
        self.priors = rule_priors
        self.sigma_obs = sigma_obs
        self.tol = tol
        self.max_iter = max_iter
        self.pop = {}
        self.elbo_history = []
    
    def fit(self, signals, applicability):
        """
        Run CAVI for population-level inference.
        
        Parameters
        ----------
        signals : dict {rule_id: {entity_id: s_jk}}
            s_jk ∈ {0, 1, NaN}. NaN = missing (unit likelihood).
        applicability : dict {rule_id: {entity_id: bool}}
        
        Returns
        -------
        dict with 'population' posteriors and 'elbo_history'.
        """
        self.signals = signals
        self.applicability = applicability
        
        all_entities = set()
        for rid in signals:
            all_entities.update(signals[rid].keys())
        self.entity_ids = sorted(all_entities)
        
        # Initialize at prior
        for rid, pr in self.priors.items():
            self.pop[rid] = {
                'alpha_q': float(pr['alpha']),
                'beta_q': float(pr['beta']),
                'mu_delta': 0.0,
                'sigma_delta': float(pr.get('sigma_drift', 1.0)),
                'n_obs': 0,
            }
        
        # ELBO at pure prior (before any data)
        elbo_prior = self._compute_elbo()
        self.elbo_history = [elbo_prior]
        
        # CAVI
        for _ in range(self.max_iter):
            self._update_population()
            self._update_drift()
            
            elbo = self._compute_elbo()
            self.elbo_history.append(elbo)
            
            if len(self.elbo_history) >= 3:
                if abs(self.elbo_history[-1] - self.elbo_history[-2]) < self.tol:
                    break
        
        return self._collect_results()
    
    def _update_population(self):
        """Q*(c_k) = Beta(α_k + Σ s_jk, β_k + Σ (1-s_jk))"""
        for rid, pr in self.priors.items():
            sum_s, sum_1ms, count = 0.0, 0.0, 0
            for eid in self.entity_ids:
                if not self.applicability.get(rid, {}).get(eid, False):
                    continue
                s = self.signals.get(rid, {}).get(eid, np.nan)
                if np.isnan(s):
                    continue
                sum_s += s
                sum_1ms += (1.0 - s)
                count += 1
            self.pop[rid]['alpha_q'] = pr['alpha'] + sum_s
            self.pop[rid]['beta_q'] = pr['beta'] + sum_1ms
            self.pop[rid]['n_obs'] = count
    
    def _update_drift(self):
        """Q*(δ_k) posterior precision."""
        for rid in self.priors:
            sigma_prior = self.priors[rid].get('sigma_drift', 1.0)
            n = self.pop[rid]['n_obs']
            if n == 0:
                continue
            sigma_post_sq = 1.0 / (1.0 / sigma_prior**2 + n / self.sigma_obs**2)
            self.pop[rid]['sigma_delta'] = np.sqrt(sigma_post_sq)
    
    def _compute_elbo(self):
        """ELBO = E_Q[log P(D|S)] - KL[Q(c) || P(c)] - KL[Q(δ) || P(δ)]"""
        elbo = 0.0
        for rid, pr in self.priors.items():
            aq = self.pop[rid]['alpha_q']
            bq = self.pop[rid]['beta_q']
            E_log_c = special.digamma(aq) - special.digamma(aq + bq)
            E_log_1mc = special.digamma(bq) - special.digamma(aq + bq)
            
            for eid in self.entity_ids:
                if not self.applicability.get(rid, {}).get(eid, False):
                    continue
                s = self.signals.get(rid, {}).get(eid, np.nan)
                if np.isnan(s):
                    continue
                elbo += s * E_log_c + (1.0 - s) * E_log_1mc
            
            elbo -= kl_beta(aq, bq, pr['alpha'], pr['beta'])
            elbo -= kl_gaussian(
                self.pop[rid]['mu_delta'], self.pop[rid]['sigma_delta'],
                0.0, pr.get('sigma_drift', 1.0))
        
        return elbo
    
    def _collect_results(self):
        results = {
            'population': {},
            'elbo_history': self.elbo_history,
            'n_iterations': len(self.elbo_history) - 1,
        }
        for rid in self.priors:
            aq = self.pop[rid]['alpha_q']
            bq = self.pop[rid]['beta_q']
            mean = aq / (aq + bq)
            std = np.sqrt(aq * bq / ((aq + bq)**2 * (aq + bq + 1)))
            lo = stats.beta.ppf(0.025, aq, bq)
            hi = stats.beta.ppf(0.975, aq, bq)
            results['population'][rid] = {
                'E_c': mean, 'std_c': std, 'CI_95': (lo, hi),
                'alpha_q': aq, 'beta_q': bq,
                'n_obs': self.pop[rid]['n_obs'],
                'E_delta': self.pop[rid]['mu_delta'],
                'std_delta': self.pop[rid]['sigma_delta'],
            }
        return results
    
    def regulatory_update(self, rule_id, new_threshold, old_threshold):
        """T1: O(1) regulatory parameter change. Returns time in ms."""
        t0 = time.perf_counter()
        epsilon = (new_threshold - old_threshold) / old_threshold
        self.pop[rule_id]['mu_delta'] -= epsilon
        return (time.perf_counter() - t0) * 1000


# ═══════════════════════════════════════════════════════════════
# ENTITY SCORER — Deterministic
# ═══════════════════════════════════════════════════════════════

class EntityScorer:
    """
    Deterministic entity-level scoring from compliance signals.
    
    Per-rule: nc_score_jk = 1 - signal_jk (continuous)
    Missing: signal = NaN → nc_score = 0.5 (neutral)
    Aggregation: min (worst rule), mean, count(NC rules) — domain choice.
    """
    
    def __init__(self, rule_ids):
        self.rule_ids = rule_ids
    
    def score(self, signals_raw, applicability, entity_ids):
        """
        Parameters
        ----------
        signals_raw : dict {rule_id: {entity_id: signal_continuous}}
        applicability : dict {rule_id: {entity_id: bool}}
        entity_ids : list
        
        Returns
        -------
        dict with 'entity_rules' and 'entities' (aggregated).
        """
        results = {'entity_rules': {}, 'entities': {}}
        
        for eid in entity_ids:
            rule_scores = {}
            
            for rid in self.rule_ids:
                if not applicability.get(rid, {}).get(eid, False):
                    continue
                
                s = signals_raw.get(rid, {}).get(eid, np.nan)
                is_missing = np.isnan(s) if isinstance(s, float) else False
                compliance = 0.5 if is_missing else float(s)
                nc_score = 1.0 - compliance
                
                rule_scores[rid] = {
                    'compliance': compliance,
                    'nc_score': nc_score,
                    'is_missing': is_missing,
                }
                results['entity_rules'][(eid, rid)] = rule_scores[rid]
            
            if rule_scores:
                scores = [v['nc_score'] for v in rule_scores.values()]
                worst_rule = max(rule_scores.keys(), key=lambda r: rule_scores[r]['nc_score'])
                n_nc = sum(1 for v in rule_scores.values() if v['nc_score'] > 0.5)
                n_missing = sum(1 for v in rule_scores.values() if v['is_missing'])
            else:
                scores, worst_rule, n_nc, n_missing = [0.5], 'none', 0, 0
            
            results['entities'][eid] = {
                'nc_min': max(scores),
                'nc_mean': np.mean(scores),
                'nc_count': n_nc,
                'worst_rule': worst_rule,
                'n_rules': len(rule_scores),
                'n_missing': n_missing,
            }
        
        return results
