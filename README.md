# Marginally Consistent Redshift Assignment

## Problem statement

We have a catalogue of galaxies. Each galaxy $g$ is assigned two discrete labels:

$$
i_L(g) \in \{1, \dots, L\}
$$

and

$$
i_S(g) \in \{1, \dots, S\}.
$$

Here, $i_L$ is the position/lens-bin assignment and $i_S$ is the shear/source-bin assignment.

For each lens bin $\ell$, we are given a target redshift density:

$$
n^L_\ell(z) = p(z \mid i_L = \ell).
$$

For each source bin $s$, we are given a target redshift density:

$$
n^S_s(z) = p(z \mid i_S = s).
$$

The catalogue contains only the two bin labels $(i_L, i_S)$, not the true redshifts. The goal is to assign a true redshift $z_g$ to each galaxy such that galaxies in lens bin $\ell$ reproduce $n^L_\ell(z)$, while galaxies in source bin $s$ reproduce $n^S_s(z)$.

This is a coupling problem. The supplied $n(z)$ distributions constrain the marginals $p(z \mid i_L)$ and $p(z \mid i_S)$, but they do not uniquely determine the full conditional distribution:

$$
p(z \mid i_L, i_S).
$$

The unknown object is therefore the redshift density conditional on both catalogue labels.

---

## Mathematical formalism

Introduce a family of normalised conditional redshift densities:

$$
q_{\ell s}(z) = q(z \mid i_L = \ell, i_S = s).
$$

Given $q_{\ell s}(z)$, the redshift density implied for lens bin $\ell$ is:

$$
\hat{n}^L_\ell(z) = \sum_s p(i_S = s \mid i_L = \ell) q_{\ell s}(z).
$$

Similarly, the redshift density implied for source bin $s$ is:

$$
\hat{n}^S_s(z) = \sum_\ell p(i_L = \ell \mid i_S = s) q_{\ell s}(z).
$$

The probabilities $p(i_S = s \mid i_L = \ell)$ and $p(i_L = \ell \mid i_S = s)$ are estimated directly from the catalogue labels.

An exact solution would satisfy:

$$
n^L_\ell(z) = \hat{n}^L_\ell(z)
$$

for every lens bin $\ell$, and

$$
n^S_s(z) = \hat{n}^S_s(z)
$$

for every source bin $s$.

Equivalently, the exact constraints are:

$$
n^L_\ell(z) = \sum_s p(i_S = s \mid i_L = \ell) q_{\ell s}(z).
$$

and

$$
n^S_s(z) = \sum_\ell p(i_L = \ell \mid i_S = s) q_{\ell s}(z).
$$

---

## Learning the conditional densities

In practice, parameterise the conditional densities as $q_{\ell s}(z) = q_\theta(z \mid i_L = \ell, i_S = s)$ — for example a conditional normalising flow $z = f_\theta(u; \ell, s)$ with $u \sim p_0(u)$. Substituting $q_\theta$ for $q_{\ell s}$ in the model-implied densities $\hat{n}^L_\ell$ and $\hat{n}^S_s$ defined above gives the model-implied mixtures $\hat{n}^L_{\ell,\theta}$ and $\hat{n}^S_{s,\theta}$.

We learn $\theta$ by minimising KL-type discrepancies between the supplied $n(z)$ distributions and these model-implied mixtures. A natural forward-KL objective is:

$$
\mathcal{L}_{\mathrm{KL}}(\theta) = -\sum_\ell \int n^L_\ell(z) \log \hat{n}^L_{\ell,\theta}(z)\,dz - \sum_s \int n^S_s(z) \log \hat{n}^S_{s,\theta}(z)\,dz + \mathrm{constant},
$$

which, when the supplied $n(z)$ distributions can be sampled from, is equivalent to:

$$
\mathcal{L}_{\mathrm{KL}}(\theta) = -\sum_\ell \mathbb{E}_{z \sim n^L_\ell}\left[\log \hat{n}^L_{\ell,\theta}(z)\right] - \sum_s \mathbb{E}_{z \sim n^S_s}\left[\log \hat{n}^S_{s,\theta}(z)\right] + \mathrm{constant}.
$$

---

## Regularisation

The problem is underdetermined: the supplied $n(z)$ distributions constrain the marginals but do not uniquely specify the full conditional $p(z \mid i_L, i_S)$, so many couplings $q_{\ell s}$ reproduce the same lens and source $n(z)$.

A normalising flow, however, already carries its own inductive bias that picks a particular solution among these. The conditional density is not a free per-cell object: it comes from a single amortised conditioner network with weights shared across all $(\ell, s)$ (here, an additive lens-embedding plus source-embedding fed through a shared MLP), restricted to smooth monotonic spline transforms of a Gaussian base, and trained from a near-identity initialisation. These choices favour smooth solutions and let poorly-constrained cells borrow structure from well-constrained ones — for example an almost-empty $(\ell, s)$ cell is reconstructed from its lens-row and source-column neighbours rather than left arbitrary.

One can also add an explicit regularisation penalty in the objective, pulling each conditional toward a reference density $r(z \mid i_L = \ell, i_S = s)$:

$$
\mathcal{L}(\theta) = \sum_\ell \mathrm{KL}\left(n^L_\ell \,\|\, \hat{n}^L_{\ell,\theta}\right) + \sum_s \mathrm{KL}\left(n^S_s \,\|\, \hat{n}^S_{s,\theta}\right) + \lambda \sum_{\ell,s} \mathrm{KL}\left(q_\theta(z \mid i_L = \ell, i_S = s) \,\|\, r(z \mid i_L = \ell, i_S = s)\right).
$$

The first two terms enforce agreement with the supplied $n(z)$ distributions, while the final term selects a particular coupling, for example by favouring smoothness, maximum entropy, or proximity to a physically motivated prior.

In practice, though, we find this explicit penalty is not very useful. Since the objective is a soft weighted sum rather than a hard constraint, any non-trivial $\lambda$ degrades the marginal fit, and pulling the conditionals toward a generic reference (e.g. a broad Gaussian) mostly fights the flow's own, better-matched inductive bias — making the recovered conditionals worse, especially in sparsely-populated cells. The implementation therefore omits it and optimises the forward-KL objective alone.

---

## Solution procedure

1. Estimate $p(i_S = s \mid i_L = \ell)$ and $p(i_L = \ell \mid i_S = s)$ directly from the catalogue labels.

2. Introduce a learnable conditional density $q_\theta(z \mid i_L = \ell, i_S = s)$, represented by a conditional normalising flow, spline density, mixture model, neural density estimator, or another normalised density model.

3. Form the model-implied lens and source mixtures $\hat{n}^L_{\ell,\theta}$ and $\hat{n}^S_{s,\theta}$ as defined above, and optimise the forward-KL objective $\mathcal{L}_{\mathrm{KL}}(\theta)$. A reference density may optionally be added to select among solutions, but is not required.

4. After training, assign each galaxy $g$ a redshift by sampling:

$$
z_g \sim q_\theta(z \mid i_L(g), i_S(g)).
$$

This produces a redshift assignment whose lens-bin and source-bin ensembles approximately reproduce the supplied redshift densities, subject to the compatibility of the input $n(z)$ distributions and the expressivity of the learned conditional density.

---

## Demo / usage

A self-contained reference implementation in PyTorch is provided here:

- [`mcra.py`](mcra.py) — the implementation. The conditional density $q_\theta(z\mid i_L, i_S)$ is a conditional rational-quadratic neural spline flow ([`zuko.flows.NSF`](https://zuko.readthedocs.io/)). It contains the Smail-type demo data generator (`make_demo_data`), the flow (`ConditionalRedshiftFlow`), the model-implied lens/source mixtures, the forward-KL training loop (`train`), and plotting/metric helpers.
- [`demo.ipynb`](demo.ipynb) — an end-to-end walkthrough on a synthetic universe with **3 lens bins and 3 source bins**, all overlapping, with the lens bins at slightly lower redshift than the source bins. It generates the data, trains the flow, and shows that the model-implied marginals and the sampled-catalogue redshifts reproduce the supplied per-bin $n(z)$.

### Setup

The only extra dependency beyond PyTorch / NumPy / Matplotlib is `zuko`:

```bash
pip install zuko
```

### Run

Open and run [`demo.ipynb`](demo.ipynb) top to bottom, or use the module directly:

```python
import mcra

data = mcra.make_demo_data()                       # synthetic 3-lens / 3-source problem
model = mcra.ConditionalRedshiftFlow(
    data.n_lens, data.n_source, data.z_mean, data.z_std
)
mcra.train(model, data, mcra.TrainConfig(steps=2000))

# assign redshifts to a catalogue of (i_L, i_S) labels
i_l, i_s, _ = mcra.sample_catalogue(data, n=100_000)
z = mcra.assign_redshifts(model, i_l, i_s)
```
