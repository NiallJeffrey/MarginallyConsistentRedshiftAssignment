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

## Extension: galaxies absent from a sample

So far every galaxy carries both a lens label and a source label. In practice the two samples need not coincide: the source sample typically extends to higher redshift than the lens sample, so many galaxies appear as sources but have no lens-bin assignment (and, less often, vice versa). We model this by augmenting each label space with an explicit **absent** token $\varnothing$ (the value coded as `-1` in a catalogue):

$$
i_L \in \{1, \dots, L\} \cup \{\varnothing\}, \qquad i_S \in \{1, \dots, S\} \cup \{\varnothing\}.
$$

A galaxy is in the catalogue if it belongs to at least one sample, so every galaxy has at least one real label and the corner $(\varnothing, \varnothing)$ is unobserved and excluded:

$$
\mathcal{C} = \big(\{1, \dots, L\} \cup \{\varnothing\}\big) \times \big(\{1, \dots, S\} \cup \{\varnothing\}\big) \setminus \{(\varnothing, \varnothing)\}.
$$

The conditional family $q_{\ell s}(z)$ is extended over all observed label pairs $(\ell, s) \in \mathcal{C}$, including the **border** densities $q_{\ell \varnothing}(z)$ (galaxies in lens bin $\ell$ that are not sources) and $q_{\varnothing s}(z)$ (galaxies in source bin $s$ that are not lenses).

### Modified marginals

Crucially, the supplied targets exist **only for the real bins** — there is no supplied $n(z)$ for the absent token. The catalogue label statistics, however, are estimated over the augmented index (the absent token included), and the model-implied marginals must sum over it:

$$
\hat{n}^L_{\ell,\theta}(z) = \sum_{b \in \{1, \dots, S\} \cup \{\varnothing\}} p(i_S = b \mid i_L = \ell)\, q_\theta(z \mid i_L = \ell, i_S = b),
$$

$$
\hat{n}^S_{s,\theta}(z) = \sum_{a \in \{1, \dots, L\} \cup \{\varnothing\}} p(i_L = a \mid i_S = s)\, q_\theta(z \mid i_L = a, i_S = s).
$$

The forward-KL objective is unchanged in form but is summed over the real bins only, since these are the only bins with a supplied target:

$$
\mathcal{L}_{\mathrm{KL}}(\theta) = -\sum_{\ell=1}^{L} \mathbb{E}_{z \sim n^L_\ell}\!\left[\log \hat{n}^L_{\ell,\theta}(z)\right] - \sum_{s=1}^{S} \mathbb{E}_{z \sim n^S_s}\!\left[\log \hat{n}^S_{s,\theta}(z)\right].
$$

The border conditionals $q_{\ell\varnothing}$ and $q_{\varnothing s}$ carry no target term of their own; they are learned implicitly through the mixtures of the real bins in which they appear.

### What changes mathematically

- **Global compatibility is lost.** In the original problem the two sets of marginals shared a common total density, $\sum_\ell p(i_L = \ell)\, n^L_\ell = \sum_s p(i_S = s)\, n^S_s = n_{\mathrm{pop}}(z)$. With partial membership this no longer holds: $\sum_\ell p(i_L = \ell)\, n^L_\ell$ is the redshift density of the *lens sample* and $\sum_s p(i_S = s)\, n^S_s$ that of the *source sample*, and the two samples cover different redshift ranges. The only remaining cross-consistency is that the shared interior conditionals $q_{\ell s}$ are a single object entering both families of mixtures.

- **The border conditionals are residuals.** $q_{\ell\varnothing}$ appears only in $\hat{n}^L_\ell$. Given the interior conditionals $q_{\ell s}$ (which are pinned by the source-side constraints), it is fixed by

$$
p(i_S = \varnothing \mid i_L = \ell)\, q_{\ell\varnothing}(z) = n^L_\ell(z) - \sum_{s=1}^{S} p(i_S = s \mid i_L = \ell)\, q_{\ell s}(z).
$$

For this to define a valid density the right-hand side must be pointwise non-negative and integrate to $p(i_S = \varnothing \mid i_L = \ell)$ — a feasibility condition on the inputs. The analogous statement holds for $q_{\varnothing s}$.

- **Degenerate limits.** If $p(i_S = \varnothing \mid i_L = \ell) = 0$ (every lens-$\ell$ galaxy is also a source), the cell $q_{\ell\varnothing}$ has zero weight and is irrelevant. If a source bin $s$ has no lens overlap, $p(i_L = \varnothing \mid i_S = s) = 1$ and $q_{\varnothing s} = n^S_s$ exactly.

### Assignment

After training, every catalogued galaxy is assigned a redshift by sampling its conditional, exactly as before — including the single-sample galaxies, which draw from the inferred border conditionals:

$$
z_g \sim q_\theta\!\left(z \mid i_L(g), i_S(g)\right), \qquad (i_L(g), i_S(g)) \in \mathcal{C}.
$$

This is the practical payoff: it yields self-consistent redshifts for, for example, the high-redshift source-only galaxies that have no lens-bin counterpart.

---

## Demo / usage

A self-contained reference implementation in PyTorch is provided here:

- [`mcra.py`](mcra.py) — the implementation. The conditional density $q_\theta(z\mid i_L, i_S)$ is a conditional rational-quadratic neural spline flow ([`zuko.flows.NSF`](https://zuko.readthedocs.io/)). It contains the Smail-type demo data generator (`make_demo_data`), the flow (`ConditionalRedshiftFlow`), the model-implied lens/source mixtures, the forward-KL training loop (`train`), and plotting/metric helpers.
- [`demo.ipynb`](demo.ipynb) — an end-to-end walkthrough on a synthetic universe with **3 lens bins and 3 source bins**, all overlapping, with the lens bins at slightly lower redshift than the source bins. It generates the data, trains the flow, and shows that the model-implied marginals and the sampled-catalogue redshifts reproduce the supplied per-bin $n(z)$.
- [`demo_partial.ipynb`](demo_partial.ipynb) — the *partial sample membership* extension (see above): a universe where the lens sample is suppressed at high redshift, so a fraction of source galaxies have no lens bin. It trains with the augmented "absent" token, recovers the real-bin marginals, assigns redshifts to source-only galaxies, and inspects the inferred border conditionals.

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
