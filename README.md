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

In practice, parameterise the conditional densities as:

$$
q_\theta(z \mid i_L = \ell, i_S = s).
$$

For example, one could use a conditional normalising flow:

$$
z = f_\theta(u; \ell, s),
$$

with

$$
u \sim p_0(u).
$$

The model-implied lens redshift density is:

$$
\hat{n}^L_{\ell,\theta}(z) = \sum_s p(i_S = s \mid i_L = \ell) q_\theta(z \mid i_L = \ell, i_S = s).
$$

The model-implied source redshift density is:

$$
\hat{n}^S_{s,\theta}(z) = \sum_\ell p(i_L = \ell \mid i_S = s) q_\theta(z \mid i_L = \ell, i_S = s).
$$

We then learn $\theta$ by minimizing KL-type discrepancies between the supplied $n(z)$ distributions and the model-implied mixtures.

A natural forward-KL objective is:

$$
\mathcal{L}_{\mathrm{KL}}(\theta) = -\sum_\ell \int n^L_\ell(z) \log \hat{n}^L_{\ell,\theta}(z)\,dz - \sum_s \int n^S_s(z) \log \hat{n}^S_{s,\theta}(z)\,dz + \mathrm{constant}.
$$

If the supplied $n(z)$ distributions can be sampled from, this can be written as:

$$
\mathcal{L}_{\mathrm{KL}}(\theta) = -\sum_\ell \mathbb{E}_{z \sim n^L_\ell}\left[\log \hat{n}^L_{\ell,\theta}(z)\right] - \sum_s \mathbb{E}_{z \sim n^S_s}\left[\log \hat{n}^S_{s,\theta}(z)\right] + \mathrm{constant}.
$$

---

## Regularisation

The problem is underdetermined. The two sets of $n(z)$ distributions do not uniquely specify:

$$
p(z \mid i_L, i_S).
$$

Therefore, it is useful to add a reference density:

$$
r(z \mid i_L = \ell, i_S = s).
$$

One possible regularised objective is:

$$
\mathcal{L}(\theta) = \sum_\ell \mathrm{KL}\left(n^L_\ell \,\|\, \hat{n}^L_{\ell,\theta}\right) + \sum_s \mathrm{KL}\left(n^S_s \,\|\, \hat{n}^S_{s,\theta}\right) + \lambda \sum_{\ell,s} \mathrm{KL}\left(q_\theta(z \mid i_L = \ell, i_S = s) \,\|\, r(z \mid i_L = \ell, i_S = s)\right).
$$

The first two terms enforce agreement with the supplied lens and source $n(z)$ distributions. The final term chooses a particular solution among the many possible couplings, for example by favouring smoothness, maximum entropy, or proximity to a physically motivated prior.

---

## Solution procedure

1. Estimate $p(i_S = s \mid i_L = \ell)$ and $p(i_L = \ell \mid i_S = s)$ directly from the catalogue labels.

2. Introduce a learnable conditional density:

$$
q_\theta(z \mid i_L = \ell, i_S = s).
$$

This can be represented by a conditional normalising flow, spline density, mixture model, neural density estimator, or another normalised density model.

3. For each lens bin $\ell$, form the model-implied density:

$$
\hat{n}^L_{\ell,\theta}(z) = \sum_s p(i_S = s \mid i_L = \ell) q_\theta(z \mid i_L = \ell, i_S = s).
$$

4. For each source bin $s$, form the model-implied density:

$$
\hat{n}^S_{s,\theta}(z) = \sum_\ell p(i_L = \ell \mid i_S = s) q_\theta(z \mid i_L = \ell, i_S = s).
$$

5. Optimise the KL objective:

$$
\mathcal{L}_{\mathrm{KL}}(\theta) = -\sum_\ell \int n^L_\ell(z) \log \hat{n}^L_{\ell,\theta}(z)\,dz - \sum_s \int n^S_s(z) \log \hat{n}^S_{s,\theta}(z)\,dz + \mathrm{constant}.
$$

Optionally include regularisation toward a reference density.

6. After training, assign each galaxy $g$ a redshift by sampling:

$$
z_g \sim q_\theta(z \mid i_L(g), i_S(g)).
$$

This produces a redshift assignment whose lens-bin and source-bin ensembles approximately reproduce the supplied redshift densities, subject to the compatibility of the input $n(z)$ distributions, the expressivity of the learned conditional density, and the chosen regularisation.
