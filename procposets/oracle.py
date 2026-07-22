"""Pricing oracle for the Frank-Wolfe NPMLE.

The convex program (see npmle.py) delegates *all* combinatorial hardness to a
single subproblem: given the current mixture density d_g on groups, find the
atom theta maximising the likelihood gradient

    score(theta) = (1/G) * sum_g f_theta(g) / d_g .

At the optimum, sup_theta score = 1 (Lindsay); score - 1 is the duality gap
and hence an optimality *certificate* relative to the atom class the oracle
actually searches.  Three oracle regimes, chosen automatically:

* **Enumeration** (m <= max_exact_m): enumerate the declared poset class on
  the alphabet -- every partial order for the default general class (219 on
  m = 4, 4231 on m = 5, 130023 on m = 6), every SP order for the SP class --
  crossed with the (eps, eta, lam) grid.  The certificate is genuine global
  optimality over the declared hypothesis class *at the declared nuisance
  grids* -- grid-exact: poset-exact per grid point, silent about off-grid
  (eps, eta, lam).  Works for every noise kernel.

* **Meet-closure** (any m; general class with the uniform kernel): candidates
  are the closure of the observed distinct chains under meet, plus the empty
  order.  This regime is *also exact*, by the following reduction:

      Reduction theorem.  Fix the uniform eps-kernel and any (eps, eta).
      For an arbitrary partial order P, let S = {observed distinct traces
      that extend P} and Q = meet of the chains of S (Q = the empty order if
      S is empty).  Then Q >= P in the refinement order, so e(Q) <= e(P),
      while the extension indicator on *observed* traces is unchanged:
      every t in S extends Q by construction, and a t outside S cannot
      extend Q because Q contains P.  Since the uniform-kernel density
      p_theta(t) = (1-eta)[(1-eps) 1[t in L(P)]/e(P) + eps/m!] + eta pbar(t)
      depends on P only through the indicator (same) and 1/e(P) (no
      smaller), p_Q >= p_P pointwise on the observed traces, hence
      f_Q(g) >= f_P(g) for every group and score(Q) >= score(P).

  So the supremum of the score over *all* partial orders is attained on the
  meet-closure of the observed chains, and the duality gap computed there is
  a certificate over the full general class -- at any alphabet size, with a
  data-dependent candidate set that is typically tiny.  (The closure can in
  principle blow up; it is capped, and a truncated closure downgrades the
  certificate to lattice-restricted status, reported honestly.)  The
  reduction fails for the swap kernel, whose contamination term depends on P
  beyond the indicator, and for timed logs, whose clean density depends on P
  through the enabled-counts; those combinations fall through to the
  heuristic regime when enumeration is out of reach.

* **Heuristic lattice** (large m, swap kernel or SP class or timed logs):
  candidates from the intersection lattice of the observed group meets
  Q_g = meet(traces in g): the Q_g, their pairwise meets, meets of those
  with the Q_g again (two rounds reach every meet of <= 4 groups), and the
  observed chains (which keep the trivial mixture representable inside the
  program as a live competitor).  With the general class every candidate is
  usable; with the SP class non-SP lattice points are *skipped*, not
  repaired -- repair rules are where silent design choices breed.  This is
  the one approximate regime, and the reported gap quantifies exactly what
  it may cost.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._numerics import _log_mean_exp_rows
from .likelihood import Atom, GroupedLog, TimedGroupedLog, make_atom
from .rel import (
    IdealBudgetExceeded,
    Rel,
    get_poset_class,
    meet,
    meet_closure,
    rel_from_trace,
)


class Oracle:
    """The Frank-Wolfe pricing oracle: assemble the candidate-atom set (the regime
    is chosen automatically -- enumeration / meet-closure / heuristic lattice, see
    the module docstring) and price each atom's likelihood-gradient score against
    the current mixture density (:meth:`price`)."""

    def __init__(
        self,
        log: GroupedLog,
        eps_grid: Sequence[float],
        eta_grid: Sequence[float],
        max_exact_m: int = 6,
        noise_kernel: str = "uniform",
        poset_class="general",
        lam_grid: Sequence[float] = (1.0,),
        closure_cap: int = 200_000,
        force_regime: Optional[str] = None,
    ):
        """``force_regime`` overrides the automatic regime selection for
        benchmarking and testing: ``"lattice-heuristic"`` runs the heuristic
        lattice even where enumeration or the meet-closure would apply (the
        certificate is downgraded accordingly).  This is the declared way to
        measure what the exact regimes buy (FIGURES certificate-vs-class);
        forcing an *exact* regime out of its validity domain is refused.
        """
        if force_regime not in (None, "lattice-heuristic"):
            raise ValueError(
                f"force_regime must be None or 'lattice-heuristic', got "
                f"{force_regime!r} (exact regimes cannot be forced: their "
                f"validity conditions are checked, not asserted)"
            )
        self.log = log
        self.eps_grid = list(eps_grid)
        self.eta_grid = list(eta_grid)
        self.lam_grid = list(lam_grid)
        if noise_kernel not in ("uniform", "swap"):
            raise ValueError(
                f"unknown noise kernel {noise_kernel!r}: the declared kernels "
                f"are 'uniform' and 'swap'"
            )
        self.noise_kernel = noise_kernel
        self.poset_class = cls = get_poset_class(poset_class)
        els = frozenset(log.alphabet)
        timed = isinstance(log, TimedGroupedLog)
        if timed and noise_kernel == "swap":
            raise ValueError(
                "timed traces support only the uniform eps kernel "
                "(no mechanistic timed kernel is declared yet)"
            )

        rels = self._select_candidates(cls, timed, max_exact_m, closure_cap, force_regime)
        self._build_atoms(rels, els, cls)

    # ------------------------------------------------------------------ #

    def _select_candidates(self, cls, timed: bool, max_exact_m: int,
                           closure_cap: int, force_regime: Optional[str]) -> List[Rel]:
        """Regime dispatch: choose the candidate relation set, and set
        ``self.kind`` / ``self.exact`` (enumeration and meet-closure are exact,
        lattice-heuristic is not)."""
        log = self.log
        if force_regime == "lattice-heuristic":
            rels = self._lattice_candidates()
            self.kind = "lattice-heuristic"
        elif log.m <= max_exact_m:
            rels = cls.enumerate(log.alphabet)
            self.kind = "enumeration"
        elif (cls.contains_all_posets and cls.closed_under_meet
              and self.noise_kernel == "uniform" and not timed):
            # capability-flag dispatch: the reduction theorem needs a class
            # that contains every poset and is closed under meet -- declared
            # on the class, not encoded as pointer identity (W18)
            chains = {rel_from_trace(t) for t in log.traces}
            rels, hit_cap = meet_closure(chains, cap=closure_cap)
            if frozenset() not in rels:
                rels = [frozenset()] + rels
            self.kind = "lattice-heuristic" if hit_cap else "meet-closure"
        else:
            rels = self._lattice_candidates()
            self.kind = "lattice-heuristic"
        self.exact = self.kind in ("enumeration", "meet-closure")
        return rels

    def _build_atoms(self, rels: Iterable[Rel], els, cls) -> None:
        """Build the atom list (each candidate poset crossed with the (eps, eta,
        lam) nuisance grid) and precompute the ``(atoms x groups)`` log-density
        matrix ``self.logF``.  A candidate too wide for exact e(P) is skipped and
        downgrades a meet-closure certificate to heuristic."""
        log = self.log
        self.atoms: List[Atom] = []
        self.budget_skipped = 0
        in_L_cache = {}
        for rel in rels:
            try:
                a0 = make_atom(els, rel, 0.0, 0.0, poset_class=cls)
            except IdealBudgetExceeded:
                # too wide for exact e(P) on the declared budget: skipped, and the
                # certificate is downgraded loudly below rather than hanging (W12.1)
                self.budget_skipped += 1
                continue
            if a0 is None:
                continue  # outside the declared class (SP only): skipped by design
            in_L_cache[rel] = log.in_L(rel)
            for eps in self.eps_grid:
                for eta in self.eta_grid:
                    for lam in self.lam_grid:
                        # the declared kernel travels on every atom, eps = 0
                        # included (density-neutral there, and trace_p skips
                        # the kernel at eps = 0): a later eps line search on
                        # the atom must price under the declared kernel, not
                        # a stale "uniform" tag
                        self.atoms.append(
                            Atom(rel=rel, e=a0.e, eps=eps, eta=eta,
                                 noise_kernel=self.noise_kernel,
                                 desc=a0.desc, lam=lam)
                        )
        if self.budget_skipped and self.kind == "meet-closure":
            # the exactness claim requires *every* closure candidate priced;
            # a budget-skipped candidate downgrades the certificate honestly
            self.kind = "lattice-heuristic"
            self.exact = False
        self.logF = np.stack(
            [log.group_logf(a, in_L_cache[a.rel]) for a in self.atoms]
        )  # (A, G)

    # ------------------------------------------------------------------ #

    def _lattice_candidates(self) -> List[Rel]:
        log = self.log
        qs = {meet(*(rel_from_trace(t) for t in g)) for g in log.groups}
        cands = set(qs)
        cands |= {rel_from_trace(t) for t in log.traces}  # observed chains
        cands.add(frozenset())  # empty order: full support, so no group can
        #                         be priced at zero by every candidate
        qs = list(qs)
        for r1, r2 in combinations(qs, 2):               # round 1: pairwise meets
            cands.add(meet(r1, r2))
        round1 = list(cands)
        for r1 in round1:                                 # round 2: meets with Q_g
            for r2 in qs:
                cands.add(meet(r1, r2))
        return sorted(cands, key=lambda r: (len(r), sorted(r)))

    # ------------------------------------------------------------------ #

    def price(self, log_d: np.ndarray) -> Tuple[int, float]:
        """Best atom index and its score (1/G) sum_g f_theta(g)/d_g.

        Compared fully in log space -- log-sum-exp per atom -- so the
        ordering is exact: the previous exp(clip(.., -700, 700)).mean()
        collapsed atoms with per-group ratios beyond the clip into exact
        ties and could overflow the mean.  The score of
        the maximiser minus 1 is the Frank-Wolfe duality gap.
        """
        log_scores = _log_mean_exp_rows(self.logF - log_d[None, :])
        k = int(np.argmax(log_scores))
        with np.errstate(over="ignore"):  # a transient early-iteration score
            #                               may exceed float range: inf is
            #                               honest, the argmax is exact
            return k, float(np.exp(log_scores[k]))
