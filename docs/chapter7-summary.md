---
title: Chapter 7 summary, PCA and latent variable models
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_SVG"></script>

# Chapter 7: PCA and latent variable models, a quick recap

Short revision notes. Exercises are not covered.

## The problem: projection

Given $$N$$ objects $$\mathbf{y}_n$$ in $$M$$ dimensions, find a lower-dimensional representation $$\mathbf{x}_n$$ in $$D<M$$ dimensions that keeps what matters. Uses: visualisation, cheaper models (many models need more parameters as $$M$$ grows), feature extraction before classification or clustering. Think of a shadow of a 3D object: some projections keep much more of its shape than others.

**Variance as a proxy for "interesting".** A projection onto direction $$\mathbf{w}$$ gives $$x_n=\mathbf{w}^T\mathbf{y}_n$$. A direction where the projected data spread out a lot keeps more structure than one where everything collapses together, and if there is cluster structure, the high-variance direction tends to preserve it. So choose projections that maximise variance.

## Principal components analysis (PCA)

PCA gives $$D$$ orthonormal directions $$\mathbf{w}_1,\dots,\mathbf{w}_D$$ ($$\mathbf{w}_i^T\mathbf{w}_j=0$$ for $$i\ne j$$, $$\mathbf{w}^T\mathbf{w}=1$$), each maximising the variance subject to being orthogonal to the earlier ones. With $$D=M$$ it is just a rotation, with no information lost.

**Derivation for one direction.** Subtract the mean from the data. The variance of $$x_n=\mathbf{w}^T\mathbf{y}_n$$ is $$\sigma^2=\mathbf{w}^T\mathbf{C}\mathbf{w}$$, with $$\mathbf{C}=\frac1N\sum_n\mathbf{y}_n\mathbf{y}_n^T$$ the sample covariance. Without a constraint this grows without limit, so use a Lagrange multiplier for $$\mathbf{w}^T\mathbf{w}=1$$. Setting the derivative to zero gives

$$
\mathbf{C}\mathbf{w}=\lambda\mathbf{w},
$$

an **eigenvector equation**. Multiplying by $$\mathbf{w}^T$$ shows $$\lambda=\mathbf{w}^T\mathbf{C}\mathbf{w}$$, so the eigenvalue **is** the variance along that direction. The best direction is the eigenvector with the largest eigenvalue, the second best has the second largest, and so on.

**Recipe.**
1. Subtract the mean from every object.
2. Compute the covariance $$\mathbf{C}$$.
3. Find its eigenvectors and eigenvalues.
4. Keep the $$D$$ eigenvectors with the largest eigenvalues as the columns of $$\mathbf{W}$$.
5. Project: $$\mathbf{X}=\mathbf{Y}\mathbf{W}$$.

PCA ignores any labels (it is unsupervised), and the order of the original dimensions does not matter.

**Choosing $$D$$.** The **eigen-spectrum** (eigenvalues in decreasing order) shows how many directions carry real variance: a few large values followed by small ones suggest few interesting dimensions. The reading is subjective. If you want 2D or 3D for plotting, $$D$$ is fixed. If PCA feeds a classifier, choose $$D$$ by cross-validated performance.

**Limits.** Classical PCA assumes real-valued data and no missing values. Movie ratings (integers, mostly missing) or purchase records (binary) break it. The rest of the chapter fixes that.

## Latent variable models

A **latent variable** is a quantity that is not in the data: either a real property nobody measured, or an abstract quantity created by the model because it is useful. The PCA coordinates $$\mathbf{x}_n$$ are of the second kind. The mixture indicators $$z_{nk}$$ from Chapter 6 are latent variables too. A latent variable model says how the observed data arise from the latent quantities, then infers the latent quantities from the data.

## Variational Bayes (VB)

Another way to approximate an intractable posterior (like Laplace and MCMC in Chapter 4). Let $$\theta$$ be all unknowns (parameters and latent variables) and $$Y$$ the data. The log marginal likelihood is hard to compute or maximise. Introduce any distribution $$Q(\theta)$$ and apply Jensen's inequality:

$$
\log p(Y)\;\ge\;\mathcal{L}(Q)=\int Q(\theta)\log\frac{p(Y,\theta)}{Q(\theta)}\,d\theta .
$$

The gap is a Kullback–Leibler divergence:

$$
\log p(Y)-\mathcal{L}(Q)=\mathrm{KL}\big[Q(\theta)\,\|\,p(\theta\mid Y)\big]\;\ge 0 ,
$$

which is zero only when $$Q$$ equals the true posterior. So **maximising the bound over $$Q$$ makes $$Q$$ an approximate posterior**, and the bound itself approximates the log marginal likelihood.

**Choosing the form of $$Q$$.** Assume it factorises, $$Q(\theta)=\prod_l Q_l(\theta_l)$$. More independence means easier optimisation but a worse approximation, because the true posterior usually has dependencies. With this factorisation each best factor satisfies

$$
Q_l(\theta_l)\;\propto\;\exp\Big(\mathbb{E}_{\text{all other }Q}\{\log p(Y,\theta)\}\Big).
$$

Each factor depends on expectations under the others, so update them in turn until convergence, as in EM. Often the exponent is a linear plus quadratic function, which shows $$Q_l$$ is a Gaussian (or gamma, etc.) and its normaliser is known.

## A probabilistic model for PCA

Link observations to latent coordinates: $$\mathbf{y}_n=\mathbf{W}\mathbf{x}_n+\text{noise}$$, with $$\mathbf{W}$$ an $$M\times D$$ matrix, noise precision $$\tau$$, and

- $$p(\mathbf{x}_n)=\mathcal{N}(\mathbf{0},\mathbf{I}_D)$$, a Gaussian prior on each row of $$\mathbf{W}$$, and a gamma prior on $$\tau$$.
- Likelihood per entry: $$p(y_{nm}\mid\mathbf{w}_m,\mathbf{x}_n,\tau)=\mathcal{N}(\mathbf{w}_m^T\mathbf{x}_n,\tau^{-1})$$.

Take $$Q(\mathbf{W},\mathbf{X},\tau)=Q_\tau(\tau)\prod_nQ_{x_n}(\mathbf{x}_n)\prod_mQ_{w_m}(\mathbf{w}_m)$$. Working through the rule above:

- $$Q_{x_n}$$ is Gaussian, with covariance $$\big[\mathbf{I}+\langle\tau\rangle\sum_m\langle\mathbf{w}_m\mathbf{w}_m^T\rangle\big]^{-1}$$ and mean $$\langle\tau\rangle\boldsymbol{\Sigma}_{x}\sum_m y_{nm}\langle\mathbf{w}_m\rangle$$.
- $$Q_{w_m}$$ is Gaussian with the same structure, with the roles of $$\mathbf{x}$$ and $$\mathbf{w}$$ swapped.
- $$Q_\tau$$ is gamma.

Algorithm: initialise, then cycle through updating $$Q_{x_n}$$ for all $$n$$, $$Q_{w_m}$$ for all $$m$$, and $$Q_\tau$$, using the expectations $$\langle\mathbf{x}_n\rangle$$, $$\langle\mathbf{x}_n\mathbf{x}_n^T\rangle$$, $$\langle\mathbf{w}_m\rangle$$ and so on, until the bound stops increasing. The posterior means of $$\mathbf{x}_n$$ play the role of the PCA projections, and the posterior covariances add information PCA does not give.

## Missing values

Introduce indicators $$z_{nm}=1$$ if entry $$y_{nm}$$ is observed, and include only observed entries in the likelihood. The updates are the same except each sum runs over observed entries only. Then the covariance of $$Q_{x_n}$$ depends on $$n$$ (and that of $$Q_{w_m}$$ on $$m$$), at extra cost, but with a useful meaning: **an object with many missing values gets a wide posterior**, so the model says honestly that it does not know where the object belongs. Expected missing values (and their variance) can also be predicted from $$\langle\mathbf{w}_m\rangle^T\langle\mathbf{x}_n\rangle$$.

Alternative: treat missing entries as extra latent variables. Then the covariance no longer depends on $$n$$ (cheaper), but every object is given the same latent uncertainty regardless of how much was observed. That is fine if few values are missing and misleading if many are.

## Non-real-valued data: probit PPCA

For binary data (e.g. how MPs voted, with many abstentions) use the **probit** likelihood $$P(y_{nm}=1\mid\mathbf{w}_m,\mathbf{x}_n)=\Phi(\mathbf{w}_m^T\mathbf{x}_n)$$ (a normal cdf, squashing to $$(0,1)$$). With this likelihood the variational factors are not of a standard form. The trick: add an **auxiliary variable** $$a_{nm}\sim\mathcal{N}(\mathbf{w}_m^T\mathbf{x}_n,1)$$ and let the observation only report its sign, $$y_{nm}=+1$$ if $$a_{nm}>0$$, $$-1$$ if $$a_{nm}<0$$. Integrating out $$a_{nm}$$ gives back the probit likelihood. With $$a_{nm}$$ kept, the updates for $$\mathbf{x}_n$$ and $$\mathbf{w}_m$$ are the Gaussian ones above with $$y_{nm}$$ replaced by $$\langle a_{nm}\rangle$$ and noise variance 1. $$Q(a_{nm})$$ is a Gaussian **truncated** to positive values when $$y_{nm}=+1$$ and to negative values when $$y_{nm}=-1$$, and its mean has a closed form involving the normal pdf and cdf.

**Example (UK MPs' votes).** Plain PCA, with zeros inserted for missing votes, pulls infrequent voters toward the origin, so position mixes political leaning with attendance. Probit PPCA, which treats missing votes properly, clusters MPs by party without that pull, and small parties form their own coherent groups. Its posterior covariances show which MPs are placed with least certainty (those who rarely voted).

**A way to see it.** The model is classification without input features: each vote is a classification task, the latent positions $$\mathbf{x}_n$$ are inferred, and the $$\mathbf{w}_m$$ define one decision boundary per vote, arranged so that as many votes as possible are satisfied.

## Take-aways

1. PCA projects onto the top eigenvectors of the covariance matrix, which are the directions of greatest variance. The eigenvalue is the variance along that direction.
2. The eigen-spectrum hints at how many dimensions matter. Choose $$D$$ by purpose.
3. A latent variable model explains data through unobserved quantities, which it then infers. Mixture models and PCA are both examples.
4. Variational Bayes maximises a lower bound on the log marginal likelihood. This is the same as minimising the KL divergence from $$Q$$ to the posterior, giving an approximate posterior built from simple factors.
5. The probabilistic version of PCA handles missing values by modelling only what is observed, and reports uncertainty per object. Binary data are handled with the probit likelihood and an auxiliary variable.
