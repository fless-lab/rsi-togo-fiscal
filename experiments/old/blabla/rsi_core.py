"""
rsi_core.py
===========
Rule-State Inference (RSI) — Core mathématique, domain-agnostic.

Contient UNIQUEMENT :
  - RuleState     : état latent d'une règle (Beta × Gaussien)
  - VariationalRSI: CAVI exact, ELBO monotone non-décroissant

Aucune logique domaine ici. Zéro règle fiscale. Zéro seuil.
Pour instancier RSI sur un domaine, voir rsi_togo.py (fiscal Togo)
ou créer votre propre adaptateur en implémentant :
  - compliance_signal_fn(obs: dict, rule: RuleState) -> float in [0,1]
  - applicability_fn(obs: dict, rule: RuleState) -> bool

═══════════════════════════════════════════════════════════════════
MODÈLE GÉNÉRATIF (conditionnel sur applicabilité observée)

  c_i  ~ Beta(α_i, β_i)                  prior taux de conformité
  δ_i  ~ N(0, σ²_drift)                  prior drift paramétrique
  y_j  | c_i, a_ij=1 ~ Bernoulli(c_i)   observation conformité

FAMILLE VARIATIONNELLE (mean-field, 2 facteurs)

  Q(S_i) = Q(c_i) · Q(δ_i)
  Q(c_i) = Beta(ᾱ_i, β̄_i)
  Q(δ_i) = N(μ̄_i, σ̄²_i)

UPDATES CAVI EXACTS

  ᾱ_i = α_i + Σ_{j∈App_i} s_j          (conjugué Beta-Bernoulli)
  β̄_i = β_i + Σ_{j∈App_i} (1 - s_j)
  σ̄²_i = 1 / (1/σ²_drift + n/σ²_obs)   (conjugué Gaussien)
  μ̄_i  = σ̄²_i · Σ δ̂_j / σ²_obs

ELBO EXACT (monotone non-décroissant par construction CAVI)

  ELBO = Σ_i [E_Q[logP(D_i|c_i)] − KL[Q(c_i)‖P(c_i)] − KL[Q(δ_i)‖P(δ_i)]]

═══════════════════════════════════════════════════════════════════
Référence :
  Atarmla, A-R. (2026). Rule-State Inference (RSI): A Bayesian
  Framework for Compliance Monitoring in Rule-Governed Domains.
"""

import numpy as np
from scipy.special import digamma, betaln
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# RULE STATE
# ══════════════════════════════════════════════════════════════════

@dataclass
class RuleState:
    """
    État latent d'une règle réglementaire r_i.

    Prior    : Beta(α_i, β_i) × N(0, σ²_drift)
    Posterior: Beta(ᾱ_i, β̄_i) × N(μ̄_i, σ̄²_i)

    Parameters
    ----------
    rule_id : str
        Identifiant unique de la règle.
    description : str
        Description lisible.
    alpha, beta : float
        Hyperparamètres du prior Beta sur le taux de conformité c_i.
        E[c_i] = alpha / (alpha + beta)
    sigma_drift : float
        Écart-type du prior gaussien sur le drift paramétrique δ_i.
    threshold : float, optional
        Seuil réglementaire de référence θ_i (ex : 60M FCFA pour TVA).
        Utilisé pour calculer le drift δ̂_j = (x_j − θ_i) / θ_i.
    """
    rule_id:       str
    description:   str
    alpha:         float = 5.0
    beta:          float = 2.0
    sigma_drift:   float = 1.0
    threshold:     Optional[float] = None

    # Paramètres variationnels — initialisés au prior
    q_alpha:       float = field(default=None)
    q_beta:        float = field(default=None)
    q_mu_drift:    float = 0.0
    q_sigma_drift: float = field(default=None)

    def __post_init__(self):
        if self.q_alpha is None:       self.q_alpha = self.alpha
        if self.q_beta  is None:       self.q_beta  = self.beta
        if self.q_sigma_drift is None: self.q_sigma_drift = self.sigma_drift

    # ── Quantités analytiques ─────────────────────────────────────

    def E_c(self) -> float:
        """E_Q[c_i] = ᾱ / (ᾱ + β̄)"""
        return self.q_alpha / (self.q_alpha + self.q_beta)

    def Var_c(self) -> float:
        """Var_Q[c_i] = ᾱβ̄ / [(ᾱ+β̄)²(ᾱ+β̄+1)]"""
        a, b = self.q_alpha, self.q_beta
        s = a + b
        return (a * b) / (s * s * (s + 1.0))

    def std_c(self) -> float:
        """std_Q[c_i] — décroît en 1/√N (Bernstein-von Mises, T2)"""
        return float(np.sqrt(self.Var_c()))

    def E_log_c(self) -> float:
        """E_Q[log c_i] = ψ(ᾱ) − ψ(ᾱ+β̄)"""
        return float(digamma(self.q_alpha) - digamma(self.q_alpha + self.q_beta))

    def E_log_1mc(self) -> float:
        """E_Q[log(1−c_i)] = ψ(β̄) − ψ(ᾱ+β̄)"""
        return float(digamma(self.q_beta) - digamma(self.q_alpha + self.q_beta))

    def expected_ell(self, signals: List[float]) -> float:
        """
        E_Q[log P(D_i | c_i)] = Σ_j [s_j · E[log c] + (1-s_j) · E[log(1-c)]]
        Somme sur les observations applicables uniquement.
        """
        if not signals:
            return 0.0
        s = np.array(signals, dtype=float)
        return float(np.sum(s * self.E_log_c() + (1.0 - s) * self.E_log_1mc()))

    def kl_beta(self) -> float:
        """KL[Beta(ᾱ,β̄) ‖ Beta(α,β)] — exact, ≥ 0"""
        qa, qb, pa, pb = self.q_alpha, self.q_beta, self.alpha, self.beta
        return float(
            betaln(pa, pb) - betaln(qa, qb)
            + (qa - pa) * digamma(qa)
            + (qb - pb) * digamma(qb)
            + (pa - qa + pb - qb) * digamma(qa + qb)
        )

    def kl_gaussian(self) -> float:
        """KL[N(μ̄,σ̄²) ‖ N(0,σ²_drift)] — exact, ≥ 0"""
        return float(
            np.log(self.sigma_drift / (self.q_sigma_drift + 1e-10))
            + (self.q_sigma_drift**2 + self.q_mu_drift**2) / (2.0 * self.sigma_drift**2)
            - 0.5
        )

    # ── Théorème T1 : update réglementaire O(1) ──────────────────

    def update_regulatory_params(self, new_threshold: float) -> dict:
        """
        T1 — Mise à jour O(1) lors d'un changement de seuil θ_i → θ'_i.

        Dérivation (1er ordre) :
            δ̂_new = (x − θ') / θ' = δ̂_old − ε  où ε = (θ' − θ) / θ
        ⟹  μ̄' = μ̄ − ε     (shift du drift)
            σ̄' = σ̄           INVARIANT
            ᾱ, β̄             INVARIANTS

        Coût : O(1). Aucune ré-inférence.
        """
        if self.threshold is None or self.threshold == 0.0:
            self.threshold = new_threshold
            return {'relative_shift': 0.0, 'cost': 'O(1)'}

        eps = (new_threshold - self.threshold) / self.threshold
        self.q_mu_drift -= eps
        self.threshold   = new_threshold

        return {
            'old_threshold':  self.threshold if self.threshold != new_threshold else new_threshold - eps * self.threshold,
            'new_threshold':  new_threshold,
            'relative_shift': eps,
            'new_q_mu_drift': self.q_mu_drift,
            'q_sigma_drift':  self.q_sigma_drift,  # invariant
            'q_alpha':        self.q_alpha,          # invariant
            'q_beta':         self.q_beta,           # invariant
            'cost':           'O(1)',
        }

    def reset(self):
        """Réinitialise le posterior au prior."""
        self.q_alpha      = self.alpha
        self.q_beta       = self.beta
        self.q_mu_drift   = 0.0
        self.q_sigma_drift = self.sigma_drift

    def posterior_summary(self) -> dict:
        """Résumé du posterior sous forme de dict."""
        return {
            'E[c]':       round(self.E_c(), 5),
            'std[c]':     round(self.std_c(), 6),
            'E[delta]':   round(self.q_mu_drift, 5),
            'std[delta]': round(self.q_sigma_drift, 5),
            'alert':      self.E_c() < 0.5,
        }


# ══════════════════════════════════════════════════════════════════
# VARIATIONAL RSI — CAVI EXACT
# ══════════════════════════════════════════════════════════════════

class VariationalRSI:
    """
    CAVI exact pour RSI (2 facteurs indépendants : Q(c_i) et Q(δ_i)).

    Note de modélisation : l'applicabilité a_{ij} est conditionnée
    (fournie par applicability_fn) et non inférée. Cette décision
    élimine la dégénérescence mean-field des modèles spike-and-slab
    dans les domaines où l'applicabilité est objectivement déterminable
    (ex : vérification de seuil légal en fiscalité).

    Parameters
    ----------
    rules : Dict[str, RuleState]
        Règles à inférer.
    compliance_signal_fn : callable
        (obs: dict, rule: RuleState) -> float in [0, 1]
        Signal de conformité fourni par l'adaptateur domaine.
        0 = non-conforme, 1 = conforme, 0.5 = neutre/manquant.
    applicability_fn : callable
        (obs: dict, rule: RuleState) -> bool
        Détermine si la règle s'applique à cette observation.
    sigma_obs : float
        Variance d'observation pour le drift gaussien σ²_obs.
    max_iter : int
        Nombre maximum d'itérations CAVI.
    tol : float
        Critère de convergence : |ΔELBO| < tol.
    """

    def __init__(
        self,
        rules: Dict[str, RuleState],
        compliance_signal_fn: Callable,
        applicability_fn: Callable,
        sigma_obs: float = 0.0625,
        max_iter: int = 100,
        tol: float = 1e-8,
    ):
        self.rules    = rules
        self.csf      = compliance_signal_fn
        self.apf      = applicability_fn
        self.sigma_obs = sigma_obs
        self.max_iter  = max_iter
        self.tol       = tol
        self.elbo_history: List[float] = []
        self._n_total: int = 0

    # ── Précomputation ────────────────────────────────────────────

    def _precompute(
        self, observations: List[dict]
    ) -> Dict[str, Tuple[List[float], List[float]]]:
        """
        Calcule les signaux de conformité et les drifts observés par règle.
        Retourne {rule_id: (signals, drifts)}.
        """
        self._n_total = len(observations)
        cache = {}
        for rid, rule in self.rules.items():
            signals, drifts = [], []
            for obs in observations:
                if not self.apf(obs, rule):
                    continue
                s = float(np.clip(self.csf(obs, rule), 1e-7, 1.0 - 1e-7))
                signals.append(s)
                if rule.threshold is not None and rule.threshold > 0.0:
                    x = obs.get('_primary_value', obs.get('obs_ca_declare', 0.0))
                    drifts.append((x - rule.threshold) / rule.threshold)
            cache[rid] = (signals, drifts)
        return cache

    # ── Updates CAVI exacts ───────────────────────────────────────

    def _update_beta(self, rule: RuleState, signals: List[float]):
        """
        Update conjugué exact Beta-Bernoulli :
            ᾱ = α + Σ s_j,   β̄ = β + Σ(1 - s_j)
        """
        if not signals:
            rule.q_alpha = rule.alpha
            rule.q_beta  = rule.beta
            return
        s = np.array(signals, dtype=float)
        rule.q_alpha = rule.alpha + float(s.sum())
        rule.q_beta  = rule.beta  + float((1.0 - s).sum())

    def _update_drift(self, rule: RuleState, drifts: List[float]):
        """
        Update conjugué exact Gaussien-Gaussien :
            σ̄² = 1 / (1/σ²_drift + n/σ²_obs)
            μ̄  = σ̄² · Σ δ̂_j / σ²_obs
        """
        if not drifts:
            return
        n = len(drifts)
        prec_prior = 1.0 / rule.sigma_drift**2
        prec_obs   = n / self.sigma_obs
        prec_post  = prec_prior + prec_obs
        rule.q_sigma_drift = float(np.sqrt(1.0 / prec_post))
        rule.q_mu_drift    = float(np.sum(drifts) / self.sigma_obs / prec_post)

    # ── ELBO exact ────────────────────────────────────────────────

    def compute_elbo(
        self, cache: Dict[str, Tuple[List[float], List[float]]]
    ) -> float:
        """
        ELBO = Σ_i [E_Q[logP(D_i|c_i)] − KL[Q(c_i)‖P(c_i)] − KL[Q(δ_i)‖P(δ_i)]]

        T3 : monotone non-décroissant par construction CAVI.
        Chaque update est le maximum global de l'ELBO par rapport à
        son facteur (conjugaison exacte) → ELBO non-décroissant. □
        """
        elbo = 0.0
        for rid, rule in self.rules.items():
            signals, _ = cache[rid]
            elbo += rule.expected_ell(signals)
            elbo -= rule.kl_beta()
            elbo -= rule.kl_gaussian()
        return float(elbo)

    # ── Fit (CAVI loop) ───────────────────────────────────────────

    def fit(
        self,
        observations: List[dict],
        verbose: bool = False,
    ) -> Dict[str, dict]:
        """
        Lance CAVI jusqu'à convergence.

        Returns
        -------
        dict : résumé du posterior par règle
            {rule_id: {'E[c]', 'std[c]', 'n_applicable', 'p_applicable', 'alert'}}
        """
        for rule in self.rules.values():
            rule.reset()

        cache = self._precompute(observations)

        # ELBO au prior (avant tout update)
        elbo_prior = self.compute_elbo(cache)
        self.elbo_history = [elbo_prior]
        prev = elbo_prior

        for it in range(self.max_iter):
            for rid, rule in self.rules.items():
                signals, drifts = cache[rid]
                self._update_beta(rule, signals)
                self._update_drift(rule, drifts)

            elbo = self.compute_elbo(cache)
            self.elbo_history.append(elbo)
            delta = elbo - prev

            if verbose:
                print(f'  iter {it+1:3d} | ELBO={elbo:12.4f} | Δ={delta:+.3e}')

            if delta < -1e-8 and it > 0:
                raise RuntimeError(
                    f'T3 VIOLATION : ELBO décroît à iter {it+1} (Δ={delta:.2e}). '
                    f'Bug dans les updates CAVI.'
                )

            if abs(delta) < self.tol and it > 0:
                if verbose:
                    print(f'  ✓ Convergé à iter {it+1}')
                break
            prev = elbo

        return self._posterior_summary(cache)

    def _posterior_summary(
        self, cache: Dict[str, Tuple[List[float], List[float]]]
    ) -> Dict[str, dict]:
        n = max(self._n_total, 1)
        return {
            rid: {
                **rule.posterior_summary(),
                'n_applicable': len(cache[rid][0]),
                'p_applicable': round(len(cache[rid][0]) / n, 4),
                'description':  rule.description,
            }
            for rid, rule in self.rules.items()
        }

    @property
    def elbo_prior(self) -> float:
        """ELBO au prior (avant tout update)."""
        return self.elbo_history[0] if self.elbo_history else float('nan')

    @property
    def elbo_posterior(self) -> float:
        """ELBO au posterior (après convergence)."""
        return self.elbo_history[-1] if self.elbo_history else float('nan')

    @property
    def elbo_gain(self) -> float:
        """Gain ELBO : posterior − prior."""
        if len(self.elbo_history) < 2:
            return float('nan')
        return self.elbo_history[-1] - self.elbo_history[0]

    @property
    def n_iterations(self) -> int:
        """Nombre d'itérations CAVI effectuées."""
        return max(0, len(self.elbo_history) - 1)

    @property
    def is_monotone(self) -> bool:
        """Vérifie T3 : ELBO non-décroissant."""
        h = self.elbo_history
        return all(h[i+1] >= h[i] - 1e-9 for i in range(len(h) - 1))
