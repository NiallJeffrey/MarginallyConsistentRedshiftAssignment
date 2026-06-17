"""Marginally Consistent Redshift Assignment.

A small, self-contained PyTorch implementation of the method described in the
project README: given a catalogue of galaxies that each carry two discrete
labels -- a lens-bin index ``i_L`` and a source-bin index ``i_S`` -- learn a
conditional redshift density ``q(z | i_L, i_S)`` (a conditional normalizing
flow) such that the label-weighted mixtures reproduce supplied per-bin target
redshift distributions ``n^L_l(z)`` and ``n^S_s(z)``.

Everything lives in this one module:

* ``Smail`` / ``SmailMixture``      -- the Smail-type redshift distributions used
                                        as the synthetic ground truth.
* ``make_demo_data``                -- builds the (synthetic) demo problem: a
                                        known coupling, the supplied per-bin
                                        targets, the catalogue label statistics,
                                        and a catalogue sampler.
* ``ConditionalRedshiftFlow``       -- a conditional rational-quadratic neural
                                        spline flow (``zuko.flows.NSF``).
* ``model_lens_log_density`` / ...  -- the model-implied lens/source mixtures.
* ``train``                         -- the forward-KL training loop.
* plotting / metric helpers         -- diagnostics used by ``demo.ipynb``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko

__all__ = [
    "Smail",
    "SmailMixture",
    "gaussian_windows",
    "SelectionDensity",
    "DemoData",
    "make_demo_data",
    "sample_catalogue",
    "ConditionalRedshiftFlow",
    "model_lens_log_density",
    "model_source_log_density",
    "TrainConfig",
    "train",
    "assign_redshifts",
    "marginal_metrics",
    "plot_marginals",
    "plot_conditionals",
    "plot_catalogue",
    # Partial sample-membership extension
    "PartialDemoData",
    "make_partial_demo",
    "sample_partial_catalogue",
    "to_model_slots",
    "plot_partial_marginals",
    "plot_partial_conditionals",
    "plot_partial_catalogue",
]


# ---------------------------------------------------------------------------
# Smail-type redshift distributions
# ---------------------------------------------------------------------------
class Smail:
    r"""A Smail-type redshift distribution.

    .. math::  n(z) \propto z^{\alpha} \exp\!\big[-(z / z_0)^{\beta}\big],

    which is a generalised gamma distribution. This gives both a closed-form
    (normalised) density and an exact sampler: with
    ``t ~ Gamma(k=(alpha+1)/beta, 1)`` we have ``z = z0 * t**(1/beta)``.
    """

    def __init__(self, z0: float, alpha: float = 2.0, beta: float = 1.5):
        self.z0 = float(z0)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = (self.alpha + 1.0) / self.beta
        # log normalisation constant of  z**alpha * exp(-(z/z0)**beta)
        self._log_norm = (
            math.log(self.beta)
            - (self.alpha + 1.0) * math.log(self.z0)
            - math.lgamma(self.k)
        )

    @property
    def peak(self) -> float:
        """Mode of the distribution."""
        return self.z0 * (self.alpha / self.beta) ** (1.0 / self.beta)

    def moment(self, m: int) -> float:
        """``E[z**m]``."""
        return self.z0 ** m * math.exp(
            math.lgamma((self.alpha + 1.0 + m) / self.beta) - math.lgamma(self.k)
        )

    @property
    def mean(self) -> float:
        return self.moment(1)

    def log_pdf(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(z, dtype=torch.float32)
        return self._log_norm + self.alpha * torch.log(z) - (z / self.z0) ** self.beta

    def pdf(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_pdf(z))

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        gamma = torch.distributions.Gamma(
            torch.tensor(self.k), torch.tensor(1.0)
        )
        # torch.distributions does not accept a generator; draw via the rsample
        # transform of a manually-seeded uniform-free gamma is overkill here, so
        # we simply use the global RNG (seeded by ``train``/``make_demo_data``).
        t = gamma.sample((n,))
        return self.z0 * t ** (1.0 / self.beta)


class SmailMixture:
    """A weighted mixture of :class:`Smail` components."""

    def __init__(self, components: Sequence[Smail], weights: Sequence[float]):
        self.components = list(components)
        w = torch.as_tensor(weights, dtype=torch.float32)
        self.weights = w / w.sum()

    def log_pdf(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(z, dtype=torch.float32)
        log_comp = torch.stack([c.log_pdf(z) for c in self.components], dim=-1)
        log_w = torch.log(self.weights)
        return torch.logsumexp(log_comp + log_w, dim=-1)

    def pdf(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_pdf(z))

    @property
    def mean(self) -> float:
        return float(sum(float(w) * c.mean for w, c in zip(self.weights, self.components)))

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.multinomial(self.weights, n, replacement=True)
        out = torch.empty(n, dtype=torch.float32)
        for k, comp in enumerate(self.components):
            mask = idx == k
            count = int(mask.sum())
            if count:
                out[mask] = comp.sample(count)
        return out


def gaussian_windows(z: torch.Tensor, centers: Sequence[float], width: float) -> torch.Tensor:
    """Soft tomographic selection ``p(bin | z)`` from Gaussian windows.

    Returns a ``[..., K]`` tensor that sums to one over the ``K`` bins, so it is
    a proper categorical distribution over bin index given redshift.
    """
    z = torch.as_tensor(z, dtype=torch.float32)
    c = torch.as_tensor(centers, dtype=torch.float32)
    logits = -0.5 * ((z.unsqueeze(-1) - c) / width) ** 2
    return torch.softmax(logits, dim=-1)


class SelectionDensity:
    """A tomographic redshift bin obtained by selecting a parent Smail ``n(z)``.

    The bin density is ``n_bin(z) = sel(z) * n_pop(z) / p_bin`` where ``sel(z)``
    is a product of soft selection probabilities ``p(lens=l | z)`` and/or
    ``p(source=s | z)``, and ``p_bin = int sel(z) n_pop(z) dz`` is the bin
    occupancy. With lens windows centred below source windows, lens bins sit at
    slightly lower redshift than source bins while overlapping.

    Additional multiplicative selection factors (callables ``z -> weight`` in
    ``[0, 1]``) can be supplied via ``factors``; these are used by the partial
    sample-membership extension to encode sample-membership probabilities
    ``pi_L(z)``, ``pi_S(z)`` and their complements.
    """

    def __init__(
        self,
        population: Smail,
        z_grid: torch.Tensor,
        lens_centers: Optional[Sequence[float]] = None,
        source_centers: Optional[Sequence[float]] = None,
        width: float = 0.12,
        l: Optional[int] = None,
        s: Optional[int] = None,
        factors: Optional[Sequence[Callable[[torch.Tensor], torch.Tensor]]] = None,
    ):
        self.population = population
        self.z_grid = z_grid
        self.lens_centers = lens_centers
        self.source_centers = source_centers
        self.width = width
        self.l = l
        self.s = s
        self.factors = list(factors) if factors is not None else []
        sel = self._selection(z_grid)
        npop = population.pdf(z_grid)
        self.p_bin = float(torch.trapezoid(sel * npop, z_grid))

    def _selection(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(z, dtype=torch.float32)
        sel = torch.ones_like(z)
        if self.l is not None:
            sel = sel * gaussian_windows(z, self.lens_centers, self.width)[..., self.l]
        if self.s is not None:
            sel = sel * gaussian_windows(z, self.source_centers, self.width)[..., self.s]
        for factor in self.factors:
            sel = sel * factor(z)
        return sel

    def pdf(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(z, dtype=torch.float32)
        return self._selection(z) * self.population.pdf(z) / self.p_bin

    def log_pdf(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log(self.pdf(z) + 1e-30)

    @property
    def mean(self) -> float:
        p = self.pdf(self.z_grid)
        return float(torch.trapezoid(self.z_grid * p, self.z_grid))

    def sample(self, n: int) -> torch.Tensor:
        """Rejection sampling: draw ``z ~ n_pop`` and accept with prob ``sel(z)``."""
        out = []
        need = n
        while need > 0:
            batch = max(int(need / max(self.p_bin, 1e-3)) + 1, 64)
            z = self.population.sample(batch)
            acc = self._selection(z)
            keep = z[torch.rand_like(z) < acc]
            out.append(keep)
            need -= int(keep.numel())
        return torch.cat(out)[:n]


# ---------------------------------------------------------------------------
# Synthetic demo problem
# ---------------------------------------------------------------------------
@dataclass
class DemoData:
    """Container for the synthetic demo problem.

    The *only* quantities a solver is allowed to use are the supplied per-bin
    targets (:attr:`lens_targets`, :attr:`source_targets`) and the catalogue
    label statistics (:attr:`p_s_given_l`, :attr:`p_l_given_s`). The true
    coupling :attr:`cond` and the catalogue redshifts are retained purely so we
    can check the recovered solution.
    """

    n_lens: int
    n_source: int
    population: Smail  # parent redshift distribution n(z)
    lens_centers: np.ndarray  # [L] lens selection-window centres
    source_centers: np.ndarray  # [S] source selection-window centres
    width: float  # selection-window width
    joint: np.ndarray  # [L, S] joint label probabilities p(i_L, i_S)
    p_l: np.ndarray  # [L]
    p_s: np.ndarray  # [S]
    p_s_given_l: np.ndarray  # [L, S]
    p_l_given_s: np.ndarray  # [L, S]
    cond: List[List[SelectionDensity]]  # true conditional q*_{l,s}
    lens_targets: List[SelectionDensity]  # supplied n^L_l(z)
    source_targets: List[SelectionDensity]  # supplied n^S_s(z)
    overall_marginal: Smail  # population n(z), useful as a reference
    z_mean: float
    z_std: float
    z_max: float
    z_grid: torch.Tensor = field(repr=False, default=None)


def make_demo_data(
    alpha: float = 2.0,
    beta: float = 1.5,
    pop_scale: float = 0.5,
    lens_centers: Sequence[float] = (0.35, 0.55, 0.75),
    source_centers: Sequence[float] = (0.50, 0.70, 0.90),
    width: float = 0.13,
    z_max: float = 2.5,
    n_grid: int = 4000,
) -> DemoData:
    """Build the synthetic 3-lens / 3-source demo problem.

    A parent Smail population ``n(z) = Smail(z; pop_scale, alpha, beta)`` is split
    into tomographic bins by soft Gaussian selection windows in redshift. Lens
    windows (``lens_centers``) sit below source windows (``source_centers``), so
    the supplied lens bins are at slightly lower redshift than the source bins
    while all bins overlap. Assuming the lens and source labels are conditionally
    independent given ``z`` gives the true conditional
    ``q*_{l,s}(z) = p(l|z) p(s|z) n(z) / p(l,s)`` and the supplied targets
    ``n^L_l(z) = p(l|z) n(z) / p(l)``, ``n^S_s(z) = p(s|z) n(z) / p(s)``. These
    are mutually compatible by construction since
    ``sum_l p(l) n^L_l = sum_s p(s) n^S_s = n(z)`` (a perfect solution exists).
    """
    lens_centers = np.asarray(lens_centers, dtype=float)
    source_centers = np.asarray(source_centers, dtype=float)
    n_lens = len(lens_centers)
    n_source = len(source_centers)

    population = Smail(pop_scale, alpha, beta)
    z_grid = torch.linspace(1e-3, z_max, n_grid)

    def selection(l=None, s=None):
        return SelectionDensity(
            population, z_grid,
            lens_centers=lens_centers, source_centers=source_centers,
            width=width, l=l, s=s,
        )

    lens_targets = [selection(l=l) for l in range(n_lens)]
    source_targets = [selection(s=s) for s in range(n_source)]
    cond = [[selection(l=l, s=s) for s in range(n_source)] for l in range(n_lens)]

    p_l = np.array([t.p_bin for t in lens_targets], dtype=float)
    p_s = np.array([t.p_bin for t in source_targets], dtype=float)
    joint = np.array([[cond[l][s].p_bin for s in range(n_source)] for l in range(n_lens)], dtype=float)
    joint = joint / joint.sum()
    p_l = p_l / p_l.sum()
    p_s = p_s / p_s.sum()
    p_s_given_l = joint / joint.sum(axis=1, keepdims=True)
    p_l_given_s = joint / joint.sum(axis=0, keepdims=True)

    z_mean = population.moment(1)
    z_std = math.sqrt(max(population.moment(2) - z_mean ** 2, 1e-6))

    return DemoData(
        n_lens=n_lens,
        n_source=n_source,
        population=population,
        lens_centers=lens_centers,
        source_centers=source_centers,
        width=width,
        joint=joint,
        p_l=p_l,
        p_s=p_s,
        p_s_given_l=p_s_given_l,
        p_l_given_s=p_l_given_s,
        cond=cond,
        lens_targets=lens_targets,
        source_targets=source_targets,
        overall_marginal=population,
        z_mean=z_mean,
        z_std=z_std,
        z_max=z_max,
        z_grid=z_grid,
    )


def sample_catalogue(data: DemoData, n: int, seed: Optional[int] = None):
    """Draw ``n`` galaxies ``(i_L, i_S, z)`` from the true model.

    Sample ``z ~ n(z)`` from the parent population, then draw the lens and source
    labels from the redshift-dependent selection windows ``p(l|z)``, ``p(s|z)``.
    Returns ``(i_l, i_s, z)`` as numpy arrays; the true ``z`` is only used for
    validation, never by the solver.
    """
    if seed is not None:
        torch.manual_seed(seed)
    z = data.population.sample(n)
    p_l_given_z = gaussian_windows(z, data.lens_centers, data.width)
    p_s_given_z = gaussian_windows(z, data.source_centers, data.width)
    i_l = torch.multinomial(p_l_given_z, 1).squeeze(-1)
    i_s = torch.multinomial(p_s_given_z, 1).squeeze(-1)
    return i_l.numpy().astype(np.int64), i_s.numpy().astype(np.int64), z.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Extension: partial sample membership
# ---------------------------------------------------------------------------
# A galaxy may belong to only one sample. We augment each label space with an
# "absent" token. Internally the absent token is encoded as the *last* slot, so
# real lens bins keep their indices 0..L-1 with absent = L, and real source bins
# keep 0..S-1 with absent = S. A flow with n_lens=L+1, n_source=S+1 then covers
# the augmented grid. Public catalogue arrays use -1 for the absent token.
@dataclass
class PartialDemoData:
    """Synthetic demo where some galaxies are absent from one sample.

    The solver may use the real-bin targets (:attr:`lens_targets`,
    :attr:`source_targets`) and the catalogue label statistics
    (:attr:`p_s_given_l` of shape ``[L, S+1]`` and :attr:`p_l_given_s` of shape
    ``[L+1, S]``, both including the absent token). The true conditionals
    :attr:`cond` (an ``(L+1) x (S+1)`` grid with the ``(absent, absent)`` corner
    set to ``None``) are kept only to check the recovered solution.
    """

    n_lens: int  # number of real lens bins, L
    n_source: int  # number of real source bins, S
    lens_absent_slot: int  # = L
    source_absent_slot: int  # = S
    population: Smail
    lens_centers: np.ndarray
    source_centers: np.ndarray
    width: float
    pi_lens: Callable[[torch.Tensor], torch.Tensor]  # P(in lens sample | z)
    pi_source: Callable[[torch.Tensor], torch.Tensor]  # P(in source sample | z)
    p_s_given_l: np.ndarray  # [L, S+1], last column = absent source
    p_l_given_s: np.ndarray  # [L+1, S], last row = absent lens
    cond: List[List[Optional[SelectionDensity]]]  # [L+1][S+1], corner None
    lens_targets: List[SelectionDensity]  # length L
    source_targets: List[SelectionDensity]  # length S
    lens_bins: List[int]  # real lens bin indices, range(L)
    source_bins: List[int]  # real source bin indices, range(S)
    z_mean: float
    z_std: float
    z_max: float
    z_grid: torch.Tensor = field(repr=False, default=None)


def _sigmoid_drop(z_cut: float, w: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Membership probability that is ~1 at low z and falls to ~0 above z_cut."""
    return lambda z: torch.sigmoid(-(torch.as_tensor(z, dtype=torch.float32) - z_cut) / w)


def _sigmoid_rise(z_cut: float, w: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Membership probability that is ~0 at low z and rises to ~1 above z_cut."""
    return lambda z: torch.sigmoid((torch.as_tensor(z, dtype=torch.float32) - z_cut) / w)


def make_partial_demo(
    alpha: float = 2.0,
    beta: float = 1.5,
    pop_scale: float = 0.55,
    lens_centers: Sequence[float] = (0.30, 0.45, 0.60),
    source_centers: Sequence[float] = (0.50, 0.80, 1.10),
    width: float = 0.13,
    lens_membership: Sequence[float] = (0.70, 0.10),
    source_membership: Sequence[float] = (0.30, 0.15),
    z_max: float = 2.5,
    n_grid: int = 4000,
) -> PartialDemoData:
    """Build a synthetic demo with partial sample membership.

    A parent Smail population ``n(z)`` is split into a lens sample (present with
    probability ``pi_L(z)``, high at low z and dropping above ``lens_membership``
    cutoff) and a source sample (present with probability ``pi_S(z)``, rising with
    z). Within each sample, soft Gaussian windows define the tomographic bins. A
    galaxy is catalogued if it belongs to at least one sample; high-redshift
    galaxies are sources with no lens bin, and the lowest-redshift galaxies may be
    lenses with no source bin. Membership and bin assignment are independent given
    ``z``, so the true conditionals are

    ``q*_{l,s} ~ pi_L w^L_l . pi_S w^S_s . n``,
    ``q*_{l,absent} ~ pi_L w^L_l . (1 - pi_S) . n``,
    ``q*_{absent,s} ~ (1 - pi_L) . pi_S w^S_s . n``,

    and the supplied real-bin targets are ``n^L_l ~ pi_L w^L_l n`` and
    ``n^S_s ~ pi_S w^S_s n``.
    """
    lens_centers = np.asarray(lens_centers, dtype=float)
    source_centers = np.asarray(source_centers, dtype=float)
    L = len(lens_centers)
    S = len(source_centers)

    population = Smail(pop_scale, alpha, beta)
    z_grid = torch.linspace(1e-3, z_max, n_grid)

    pi_L = _sigmoid_drop(*lens_membership)
    pi_S = _sigmoid_rise(*source_membership)

    def w_L(l):
        return lambda z: gaussian_windows(z, lens_centers, width)[..., l]

    def w_S(s):
        return lambda z: gaussian_windows(z, source_centers, width)[..., s]

    def comp_pi_S(z):
        return 1.0 - pi_S(z)

    def comp_pi_L(z):
        return 1.0 - pi_L(z)

    def density(factors):
        return SelectionDensity(population, z_grid, factors=factors)

    # Real-bin targets: n^L_l ~ pi_L w^L_l n,  n^S_s ~ pi_S w^S_s n.
    lens_targets = [density([pi_L, w_L(l)]) for l in range(L)]
    source_targets = [density([pi_S, w_S(s)]) for s in range(S)]

    # True conditionals on the augmented (L+1) x (S+1) grid (corner excluded).
    cond: List[List[Optional[SelectionDensity]]] = [
        [None for _ in range(S + 1)] for _ in range(L + 1)
    ]
    for l in range(L):
        for s in range(S):
            cond[l][s] = density([pi_L, w_L(l), pi_S, w_S(s)])  # interior
        cond[l][S] = density([pi_L, w_L(l), comp_pi_S])  # lens-only border
    for s in range(S):
        cond[L][s] = density([comp_pi_L, pi_S, w_S(s)])  # source-only border
    # cond[L][S] stays None: galaxies in neither sample are unobserved.

    # Co-occurrence including the absent token, normalised within each real bin.
    p_s_given_l = np.zeros((L, S + 1), dtype=float)
    for l in range(L):
        denom = lens_targets[l].p_bin  # = int pi_L w^L_l n
        for s in range(S):
            p_s_given_l[l, s] = cond[l][s].p_bin / denom
        p_s_given_l[l, S] = cond[l][S].p_bin / denom

    p_l_given_s = np.zeros((L + 1, S), dtype=float)
    for s in range(S):
        denom = source_targets[s].p_bin  # = int pi_S w^S_s n
        for l in range(L):
            p_l_given_s[l, s] = cond[l][s].p_bin / denom
        p_l_given_s[L, s] = cond[L][s].p_bin / denom

    z_mean = population.moment(1)
    z_std = math.sqrt(max(population.moment(2) - z_mean ** 2, 1e-6))

    return PartialDemoData(
        n_lens=L,
        n_source=S,
        lens_absent_slot=L,
        source_absent_slot=S,
        population=population,
        lens_centers=lens_centers,
        source_centers=source_centers,
        width=width,
        pi_lens=pi_L,
        pi_source=pi_S,
        p_s_given_l=p_s_given_l,
        p_l_given_s=p_l_given_s,
        cond=cond,
        lens_targets=lens_targets,
        source_targets=source_targets,
        lens_bins=list(range(L)),
        source_bins=list(range(S)),
        z_mean=z_mean,
        z_std=z_std,
        z_max=z_max,
        z_grid=z_grid,
    )


def sample_partial_catalogue(data: PartialDemoData, n: int, seed: Optional[int] = None):
    """Draw galaxies ``(i_L, i_S, z)`` from the partial-membership model.

    Sample ``z ~ n(z)``; with probability ``pi_L(z)`` the galaxy is in the lens
    sample and gets a lens bin (else ``-1``), and likewise for the source sample.
    Galaxies in neither sample are discarded (unobserved). The returned label
    arrays use ``-1`` for the absent token. ``n`` is the number of *parent* draws;
    the returned catalogue is slightly smaller after discarding (absent, absent).
    """
    if seed is not None:
        torch.manual_seed(seed)
    z = data.population.sample(n)

    in_lens = torch.rand(n) < data.pi_lens(z)
    in_source = torch.rand(n) < data.pi_source(z)

    lbin = torch.multinomial(gaussian_windows(z, data.lens_centers, data.width), 1).squeeze(-1)
    sbin = torch.multinomial(gaussian_windows(z, data.source_centers, data.width), 1).squeeze(-1)

    i_l = torch.where(in_lens, lbin, torch.full_like(lbin, -1))
    i_s = torch.where(in_source, sbin, torch.full_like(sbin, -1))

    keep = in_lens | in_source
    return (
        i_l[keep].numpy().astype(np.int64),
        i_s[keep].numpy().astype(np.int64),
        z[keep].numpy().astype(np.float32),
    )


def to_model_slots(i_l: np.ndarray, i_s: np.ndarray, data: PartialDemoData):
    """Map public catalogue labels (``-1`` = absent) to model slot indices.

    Absent lens labels map to slot ``L`` and absent source labels to slot ``S``,
    matching the flow built with ``n_lens=L+1``, ``n_source=S+1``.
    """
    i_l = np.asarray(i_l).copy()
    i_s = np.asarray(i_s).copy()
    i_l[i_l < 0] = data.lens_absent_slot
    i_s[i_s < 0] = data.source_absent_slot
    return i_l, i_s


# ---------------------------------------------------------------------------
# Conditional spline flow
# ---------------------------------------------------------------------------
class ConditionalRedshiftFlow(nn.Module):
    """Conditional neural spline flow for ``q_theta(z | i_L, i_S)``.

    Wraps :class:`zuko.flows.NSF` (monotonic rational-quadratic splines). The
    flow operates on a standardised variable ``u = (z - mu0) / sigma0`` (the NSF
    spline domain is ``[-5, 5]``), and densities are mapped back to ``z``-space
    via the constant Jacobian ``log q(z) = log q_u(u) - log(sigma0)``. The bin
    labels are supplied to the flow as a concatenated one-hot context vector.
    """

    def __init__(
        self,
        n_lens: int,
        n_source: int,
        z_mean: float,
        z_std: float,
        bins: int = 8,
        transforms: int = 3,
        hidden_features: Sequence[int] = (64, 64),
    ):
        super().__init__()
        self.n_lens = n_lens
        self.n_source = n_source
        self.register_buffer("mu0", torch.tensor(float(z_mean)))
        self.register_buffer("sigma0", torch.tensor(float(z_std)))
        self.flow = zuko.flows.NSF(
            features=1,
            context=n_lens + n_source,
            bins=bins,
            transforms=transforms,
            hidden_features=list(hidden_features),
        )

    def context(self, i_l: torch.Tensor, i_s: torch.Tensor) -> torch.Tensor:
        cl = F.one_hot(i_l, self.n_lens).float()
        cs = F.one_hot(i_s, self.n_source).float()
        return torch.cat([cl, cs], dim=-1)

    def _standardise(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mu0) / self.sigma0

    def _unstandardise(self, u: torch.Tensor) -> torch.Tensor:
        return u * self.sigma0 + self.mu0

    def log_prob(self, z: torch.Tensor, i_l: torch.Tensor, i_s: torch.Tensor) -> torch.Tensor:
        """``log q_theta(z | i_L, i_S)`` evaluated in ``z``-space. ``z`` is ``[N]``."""
        u = self._standardise(z)
        ctx = self.context(i_l, i_s)
        log_q_u = self.flow(ctx).log_prob(u.unsqueeze(-1))
        return log_q_u - torch.log(self.sigma0)

    @torch.no_grad()
    def sample(self, n: int, i_l: int, i_s: int) -> torch.Tensor:
        """Draw ``n`` redshifts from ``q_theta(z | i_L=i_l, i_S=i_s)``."""
        device = self.mu0.device
        ctx = self.context(
            torch.tensor(i_l, device=device),
            torch.tensor(i_s, device=device),
        )
        u = self.flow(ctx).sample((n,)).squeeze(-1)
        return self._unstandardise(u)


# ---------------------------------------------------------------------------
# Model-implied mixtures
# ---------------------------------------------------------------------------
def model_lens_log_density(
    model: ConditionalRedshiftFlow, z: torch.Tensor, l: int, p_s_given_l: np.ndarray
) -> torch.Tensor:
    r"""``log \hat n^L_{l,theta}(z) = log sum_s p(s|l) q_theta(z | l, s)``."""
    device = z.device
    log_w = torch.log(torch.as_tensor(p_s_given_l[l], dtype=torch.float32, device=device))
    terms = []
    for s in range(model.n_source):
        i_l = torch.full_like(z, l, dtype=torch.long)
        i_s = torch.full_like(z, s, dtype=torch.long)
        terms.append(model.log_prob(z, i_l, i_s) + log_w[s])
    return torch.logsumexp(torch.stack(terms, dim=-1), dim=-1)


def model_source_log_density(
    model: ConditionalRedshiftFlow, z: torch.Tensor, s: int, p_l_given_s: np.ndarray
) -> torch.Tensor:
    r"""``log \hat n^S_{s,theta}(z) = log sum_l p(l|s) q_theta(z | l, s)``."""
    device = z.device
    log_w = torch.log(torch.as_tensor(p_l_given_s[:, s], dtype=torch.float32, device=device))
    terms = []
    for l in range(model.n_lens):
        i_l = torch.full_like(z, l, dtype=torch.long)
        i_s = torch.full_like(z, s, dtype=torch.long)
        terms.append(model.log_prob(z, i_l, i_s) + log_w[l])
    return torch.logsumexp(torch.stack(terms, dim=-1), dim=-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _lens_bins(model, data) -> List[int]:
    """Real lens bins that carry a supplied target (all of them unless the data
    object declares a subset, as in the partial-membership extension)."""
    bins = getattr(data, "lens_bins", None)
    return list(bins) if bins is not None else list(range(model.n_lens))


def _source_bins(model, data) -> List[int]:
    bins = getattr(data, "source_bins", None)
    return list(bins) if bins is not None else list(range(model.n_source))


@dataclass
class TrainConfig:
    steps: int = 4000
    batch_size: int = 1024
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 0
    log_every: int = 250


def train(
    model: ConditionalRedshiftFlow,
    data: DemoData,
    config: Optional[TrainConfig] = None,
    verbose: bool = True,
    **kwargs,
):
    """Fit the flow with the forward-KL objective from the README.

    Minimises (Monte-Carlo estimate of)

    ``- sum_l E_{z~n^L_l}[log nhat^L_l(z)] - sum_s E_{z~n^S_s}[log nhat^S_s(z)]``

    by sampling minibatches from the supplied per-bin targets and pushing the
    model-implied lens/source mixtures toward them. Returns a ``history`` dict of
    loss curves.
    """
    if config is None:
        config = TrainConfig(**kwargs)
    elif kwargs:
        config = TrainConfig(**{**config.__dict__, **kwargs})

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = {"loss": [], "lens_kl": [], "source_kl": []}

    lens_bins = _lens_bins(model, data)
    source_bins = _source_bins(model, data)

    for step in range(config.steps):
        opt.zero_grad()

        lens_loss = torch.zeros((), device=device)
        for li, l in enumerate(lens_bins):
            z = data.lens_targets[li].sample(config.batch_size).to(device)
            lens_loss = lens_loss - model_lens_log_density(model, z, l, data.p_s_given_l).mean()

        source_loss = torch.zeros((), device=device)
        for si, s in enumerate(source_bins):
            z = data.source_targets[si].sample(config.batch_size).to(device)
            source_loss = source_loss - model_source_log_density(model, z, s, data.p_l_given_s).mean()

        loss = lens_loss + source_loss
        loss.backward()
        opt.step()

        loss_v = float(loss.detach())
        lens_v = float(lens_loss.detach())
        source_v = float(source_loss.detach())
        history["loss"].append(loss_v)
        history["lens_kl"].append(lens_v)
        history["source_kl"].append(source_v)

        if verbose and (step % config.log_every == 0 or step == config.steps - 1):
            print(
                f"step {step:5d} | loss {loss_v:8.4f} | "
                f"lens {lens_v:8.4f} | source {source_v:8.4f}"
            )

    return history


# ---------------------------------------------------------------------------
# Redshift assignment + metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def assign_redshifts(
    model: ConditionalRedshiftFlow, i_l: np.ndarray, i_s: np.ndarray
) -> np.ndarray:
    """Assign a redshift to each catalogue galaxy by sampling ``q_theta``."""
    i_l = np.asarray(i_l)
    i_s = np.asarray(i_s)
    z = np.empty(len(i_l), dtype=np.float32)
    for l in range(model.n_lens):
        for s in range(model.n_source):
            mask = (i_l == l) & (i_s == s)
            count = int(mask.sum())
            if count:
                z[mask] = model.sample(count, l, s).cpu().numpy()
    return z


@torch.no_grad()
def _density_on_grid(log_density_fn, zgrid: torch.Tensor) -> np.ndarray:
    return torch.exp(log_density_fn(zgrid)).cpu().numpy()


@torch.no_grad()
def marginal_metrics(model: ConditionalRedshiftFlow, data: DemoData, n_grid: int = 400):
    """Per-bin agreement between supplied targets and model-implied marginals.

    Returns a dict with forward-KL ``int n log(n/nhat) dz`` and L1
    ``int |n - nhat| dz`` for every lens and source bin.
    """
    device = next(model.parameters()).device
    zgrid = torch.linspace(1e-3, data.z_max, n_grid, device=device)
    dz = float(zgrid[1] - zgrid[0])

    out = {"lens_kl": [], "lens_l1": [], "source_kl": [], "source_l1": []}
    eps = 1e-12

    for li, l in enumerate(_lens_bins(model, data)):
        target = data.lens_targets[li].pdf(zgrid.cpu()).numpy()
        implied = _density_on_grid(
            lambda z, l=l: model_lens_log_density(model, z, l, data.p_s_given_l), zgrid
        )
        kl = float(np.sum(target * (np.log(target + eps) - np.log(implied + eps))) * dz)
        l1 = float(np.sum(np.abs(target - implied)) * dz)
        out["lens_kl"].append(kl)
        out["lens_l1"].append(l1)

    for si, s in enumerate(_source_bins(model, data)):
        target = data.source_targets[si].pdf(zgrid.cpu()).numpy()
        implied = _density_on_grid(
            lambda z, s=s: model_source_log_density(model, z, s, data.p_l_given_s), zgrid
        )
        kl = float(np.sum(target * (np.log(target + eps) - np.log(implied + eps))) * dz)
        l1 = float(np.sum(np.abs(target - implied)) * dz)
        out["source_kl"].append(kl)
        out["source_l1"].append(l1)

    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _zgrid(data: DemoData, n: int = 400):
    return torch.linspace(1e-3, data.z_max, n)


@torch.no_grad()
def plot_marginals(model: ConditionalRedshiftFlow, data: DemoData, n_grid: int = 400):
    """Supplied target vs model-implied marginal for each lens and source bin."""
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    zgrid = _zgrid(data, n_grid)
    zgrid_d = zgrid.to(device)

    lens_bins = _lens_bins(model, data)
    source_bins = _source_bins(model, data)
    n_cols = max(len(lens_bins), len(source_bins))
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), sharex=True, squeeze=False)

    for li, l in enumerate(lens_bins):
        ax = axes[0, li]
        target = data.lens_targets[li].pdf(zgrid).numpy()
        implied = _density_on_grid(
            lambda z, l=l: model_lens_log_density(model, z, l, data.p_s_given_l), zgrid_d
        )
        ax.plot(zgrid.numpy(), target, "k-", lw=2, label="target $n^L_\\ell$")
        ax.plot(zgrid.numpy(), implied, "C3--", lw=2, label="implied $\\hat n^L_\\ell$")
        ax.set_title(f"Lens bin {l}")
        ax.set_ylabel("density")
        if li == 0:
            ax.legend()

    for si, s in enumerate(source_bins):
        ax = axes[1, si]
        target = data.source_targets[si].pdf(zgrid).numpy()
        implied = _density_on_grid(
            lambda z, s=s: model_source_log_density(model, z, s, data.p_l_given_s), zgrid_d
        )
        ax.plot(zgrid.numpy(), target, "k-", lw=2, label="target $n^S_s$")
        ax.plot(zgrid.numpy(), implied, "C0--", lw=2, label="implied $\\hat n^S_s$")
        ax.set_title(f"Source bin {s}")
        ax.set_xlabel("redshift $z$")
        ax.set_ylabel("density")
        if si == 0:
            ax.legend()

    # blank any unused axes
    for row, count in ((0, len(lens_bins)), (1, len(source_bins))):
        for c in range(count, n_cols):
            axes[row, c].axis("off")

    fig.suptitle("Supplied targets vs model-implied marginals", y=1.02)
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_conditionals(model: ConditionalRedshiftFlow, data: DemoData, n_grid: int = 400):
    """3x3 grid of recovered ``q_theta(z | l, s)`` overlaid with the truth."""
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    zgrid = _zgrid(data, n_grid)
    zgrid_d = zgrid.to(device)

    fig, axes = plt.subplots(
        model.n_lens, model.n_source,
        figsize=(3.2 * model.n_source, 2.8 * model.n_lens),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    for l in range(model.n_lens):
        for s in range(model.n_source):
            ax = axes[l, s]
            true = data.cond[l][s].pdf(zgrid).numpy()
            i_l = torch.full_like(zgrid_d, l, dtype=torch.long)
            i_s = torch.full_like(zgrid_d, s, dtype=torch.long)
            learned = torch.exp(model.log_prob(zgrid_d, i_l, i_s)).cpu().numpy()
            ax.plot(zgrid.numpy(), true, "k-", lw=2, label="true $q^*$")
            ax.plot(zgrid.numpy(), learned, "C2--", lw=2, label="learned $q_\\theta$")
            ax.set_title(f"$\\ell={l},\\ s={s}$", fontsize=9)
            if l == model.n_lens - 1:
                ax.set_xlabel("$z$")
            if s == 0:
                ax.set_ylabel("density")
            if l == 0 and s == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Recovered conditional densities $q_\\theta(z\\mid\\ell,s)$", y=1.02)
    fig.tight_layout()
    return fig


def plot_catalogue(
    model: ConditionalRedshiftFlow,
    data: DemoData,
    n: int = 200_000,
    seed: int = 1,
    n_grid: int = 400,
):
    """Sample a catalogue, assign redshifts, and histogram per lens/source bin."""
    import matplotlib.pyplot as plt

    i_l, i_s, _ = sample_catalogue(data, n, seed=seed)
    z_assigned = assign_redshifts(model, i_l, i_s)
    zgrid = _zgrid(data, n_grid).numpy()

    n_cols = max(model.n_lens, model.n_source)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), sharex=True)

    for l in range(model.n_lens):
        ax = axes[0, l]
        sel = z_assigned[i_l == l]
        ax.hist(sel, bins=60, density=True, alpha=0.5, color="C3", label="assigned")
        ax.plot(zgrid, data.lens_targets[l].pdf(torch.tensor(zgrid)).numpy(), "k-", lw=2, label="target")
        ax.set_title(f"Lens bin {l}")
        ax.set_ylabel("density")
        if l == 0:
            ax.legend()

    for s in range(model.n_source):
        ax = axes[1, s]
        sel = z_assigned[i_s == s]
        ax.hist(sel, bins=60, density=True, alpha=0.5, color="C0", label="assigned")
        ax.plot(zgrid, data.source_targets[s].pdf(torch.tensor(zgrid)).numpy(), "k-", lw=2, label="target")
        ax.set_title(f"Source bin {s}")
        ax.set_xlabel("redshift $z$")
        ax.set_ylabel("density")
        if s == 0:
            ax.legend()

    for row, count in ((0, model.n_lens), (1, model.n_source)):
        for c in range(count, n_cols):
            axes[row, c].axis("off")

    fig.suptitle("Assigned-catalogue redshifts vs supplied targets", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plotting helpers for the partial sample-membership extension
# ---------------------------------------------------------------------------
def plot_partial_marginals(model: ConditionalRedshiftFlow, data: PartialDemoData, n_grid: int = 400):
    """Supplied real-bin targets vs model-implied marginals (sums over absent)."""
    return plot_marginals(model, data, n_grid=n_grid)


@torch.no_grad()
def plot_partial_conditionals(model: ConditionalRedshiftFlow, data: PartialDemoData, n_grid: int = 400):
    """Recovered ``q_theta(z | a, b)`` over the augmented grid, incl. the border
    cells with one ``absent`` label, overlaid with the known truth. The
    ``(absent, absent)`` corner is unobserved and left blank."""
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    zgrid = _zgrid(data, n_grid)
    zgrid_d = zgrid.to(device)

    nL = data.n_lens + 1  # rows incl. absent lens (last)
    nS = data.n_source + 1  # cols incl. absent source (last)
    fig, axes = plt.subplots(nL, nS, figsize=(3.0 * nS, 2.6 * nL), sharex=True, sharey=True, squeeze=False)

    def label(idx, absent_idx, kind):
        return f"{kind}=\u2205" if idx == absent_idx else f"{kind}={idx}"

    for a in range(nL):
        for b in range(nS):
            ax = axes[a, b]
            if data.cond[a][b] is None:
                ax.axis("off")
                continue
            true = data.cond[a][b].pdf(zgrid).numpy()
            i_l = torch.full_like(zgrid_d, a, dtype=torch.long)
            i_s = torch.full_like(zgrid_d, b, dtype=torch.long)
            learned = torch.exp(model.log_prob(zgrid_d, i_l, i_s)).cpu().numpy()
            border = (a == data.lens_absent_slot) or (b == data.source_absent_slot)
            ax.plot(zgrid.numpy(), true, "k-", lw=2, label="true $q^*$")
            ax.plot(zgrid.numpy(), learned, ("C1--" if border else "C2--"), lw=2, label="learned")
            ttl = f"${label(a, data.lens_absent_slot, 'l')},\\ {label(b, data.source_absent_slot, 's')}$"
            ax.set_title(ttl + ("  (border)" if border else ""), fontsize=8)
            if a == nL - 1:
                ax.set_xlabel("$z$")
            if b == 0:
                ax.set_ylabel("density")
            if a == 0 and b == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Recovered conditionals on the augmented grid ($\\varnothing$ = absent)", y=1.01)
    fig.tight_layout()
    return fig


def plot_partial_catalogue(
    model: ConditionalRedshiftFlow,
    data: PartialDemoData,
    n: int = 400_000,
    seed: int = 1,
    n_grid: int = 400,
):
    """Sample a partial catalogue, assign redshifts (incl. single-sample
    galaxies), and histogram each real bin against its supplied target."""
    import matplotlib.pyplot as plt

    i_l, i_s, _ = sample_partial_catalogue(data, n, seed=seed)
    sl, ss = to_model_slots(i_l, i_s, data)
    z_assigned = assign_redshifts(model, sl, ss)
    zgrid = _zgrid(data, n_grid).numpy()

    n_cols = max(data.n_lens, data.n_source)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), sharex=True, squeeze=False)

    for l in range(data.n_lens):
        ax = axes[0, l]
        sel = z_assigned[i_l == l]
        ax.hist(sel, bins=60, density=True, alpha=0.5, color="C3", label="assigned")
        ax.plot(zgrid, data.lens_targets[l].pdf(torch.tensor(zgrid)).numpy(), "k-", lw=2, label="target")
        ax.set_title(f"Lens bin {l}")
        ax.set_ylabel("density")
        if l == 0:
            ax.legend()

    for s in range(data.n_source):
        ax = axes[1, s]
        sel = z_assigned[i_s == s]
        ax.hist(sel, bins=60, density=True, alpha=0.5, color="C0", label="assigned")
        ax.plot(zgrid, data.source_targets[s].pdf(torch.tensor(zgrid)).numpy(), "k-", lw=2, label="target")
        ax.set_title(f"Source bin {s}")
        ax.set_xlabel("redshift $z$")
        ax.set_ylabel("density")
        if s == 0:
            ax.legend()

    for row, count in ((0, data.n_lens), (1, data.n_source)):
        for c in range(count, n_cols):
            axes[row, c].axis("off")

    n_src_only = int(np.sum((i_l < 0) & (i_s >= 0)))
    n_lens_only = int(np.sum((i_s < 0) & (i_l >= 0)))
    fig.suptitle(
        f"Partial catalogue: {n_src_only:,} source-only and {n_lens_only:,} lens-only galaxies",
        y=1.02,
    )
    fig.tight_layout()
    return fig
