---
title: Exercise 5.1, posterior of a class mean with a Gaussian prior
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_SVG"></script>

# Exercise 5.1: the posterior of a class mean under a Gaussian prior

Chapter 5 notes.

**The exercise.** The training objects of class $$c$$ are $$\mathbf{x}_1,\dots,\mathbf{x}_{N_c}$$, with $$\mathbf{x}_n\sim\mathcal{N}(\boldsymbol{\mu}_c,\boldsymbol{\Sigma}_c)$$ independently. Take $$\boldsymbol{\Sigma}_c=\mathbf{I}$$ and a Gaussian prior on $$\boldsymbol{\mu}_c$$. Find $$p(\boldsymbol{\mu}_c\mid\mathbf{X}_c)$$.

The plan is to solve the most general case first (any known covariance, any Gaussian prior) and then read the specific cases off the general answer. Write $$N=N_c$$ and drop the subscript $$c$$ while working.

## The most general case

Let $$\boldsymbol{\Sigma}$$ be any known covariance and let the prior be $$p(\boldsymbol{\mu})=\mathcal{N}(\boldsymbol{\mu}_0,\boldsymbol{\Sigma}_0)$$.

**Step 1: Bayes' rule.** Only $$\boldsymbol{\mu}$$ is unknown, so

$$p(\boldsymbol{\mu}\mid\mathbf{X})=\frac{p(\mathbf{X}\mid\boldsymbol{\mu})\,p(\boldsymbol{\mu})}{p(\mathbf{X})}\ \propto\ p(\mathbf{X}\mid\boldsymbol{\mu})\,p(\boldsymbol{\mu}).$$

The denominator does not depend on $$\boldsymbol{\mu}$$, so we only need the shape of the numerator.

**Step 2: the log of the numerator.** The objects are independent given $$\boldsymbol{\mu}$$, so the likelihood is a product of Gaussians. Keeping only terms that contain $$\boldsymbol{\mu}$$,

$$\log p(\boldsymbol{\mu}\mid\mathbf{X})=-\frac12\sum_{n=1}^{N}(\mathbf{x}_n-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}_n-\boldsymbol{\mu})-\frac12(\boldsymbol{\mu}-\boldsymbol{\mu}_0)^T\boldsymbol{\Sigma}_0^{-1}(\boldsymbol{\mu}-\boldsymbol{\mu}_0)+\text{const}.$$

**Step 3: expand and collect powers of $$\boldsymbol{\mu}$$.** Using $$\sum_n\mathbf{x}_n=N\bar{\mathbf{x}}$$ with $$\bar{\mathbf{x}}=\frac1N\sum_n\mathbf{x}_n$$,

$$\log p(\boldsymbol{\mu}\mid\mathbf{X})=-\frac12\boldsymbol{\mu}^T\left(N\boldsymbol{\Sigma}^{-1}+\boldsymbol{\Sigma}_0^{-1}\right)\boldsymbol{\mu}+\boldsymbol{\mu}^T\left(N\boldsymbol{\Sigma}^{-1}\bar{\mathbf{x}}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0\right)+\text{const}.$$

The data enter only through $$N$$ and $$\bar{\mathbf{x}}$$.

**Step 4: complete the square.** A Gaussian $$\mathcal{N}(\boldsymbol{\mu}_N,\boldsymbol{\Sigma}_N)$$ has log density $$-\frac12\boldsymbol{\mu}^T\boldsymbol{\Sigma}_N^{-1}\boldsymbol{\mu}+\boldsymbol{\mu}^T\boldsymbol{\Sigma}_N^{-1}\boldsymbol{\mu}_N+\text{const}$$. Matching the quadratic and linear terms gives

$$\boxed{\ p(\boldsymbol{\mu}\mid\mathbf{X})=\mathcal{N}(\boldsymbol{\mu}_N,\boldsymbol{\Sigma}_N),\qquad
\boldsymbol{\Sigma}_N=\left(\boldsymbol{\Sigma}_0^{-1}+N\boldsymbol{\Sigma}^{-1}\right)^{-1},\qquad
\boldsymbol{\mu}_N=\boldsymbol{\Sigma}_N\left(\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0+N\boldsymbol{\Sigma}^{-1}\bar{\mathbf{x}}\right).\ }$$

**Reading the result.**

- The posterior precision is the prior precision plus $$N$$ times the precision of one observation.
- The posterior mean is a precision-weighted average of the prior mean $$\boldsymbol{\mu}_0$$ and the sample mean $$\bar{\mathbf{x}}$$. The matrix $$\boldsymbol{\Sigma}_N$$ turns the weighted sum into a proper average (when the weights are scalars, they add up to one).
- The likelihood, as a function of $$\boldsymbol{\mu}$$, is proportional to $$\mathcal{N}(\bar{\mathbf{x}},\boldsymbol{\Sigma}/N)$$. This is the "Gaussian likelihood" the product rule combines with the prior: its mean is the maximum likelihood estimate $$\bar{\mathbf{x}}$$ (Equation 5.4) and its covariance is $$\boldsymbol{\Sigma}/N$$.

**An equivalent form.** Writing the answer as a correction to the prior mean,

$$\boldsymbol{\mu}_N=\boldsymbol{\mu}_0+\boldsymbol{\Sigma}_0\left(\boldsymbol{\Sigma}_0+\tfrac1N\boldsymbol{\Sigma}\right)^{-1}(\bar{\mathbf{x}}-\boldsymbol{\mu}_0),\qquad
\boldsymbol{\Sigma}_N=\boldsymbol{\Sigma}_0-\boldsymbol{\Sigma}_0\left(\boldsymbol{\Sigma}_0+\tfrac1N\boldsymbol{\Sigma}\right)^{-1}\boldsymbol{\Sigma}_0.$$

This follows from the matrix inversion lemma. It reads: move from the prior mean towards the data mean by a fraction that depends on how uncertain each is. (I checked both forms against each other, and that the gradient of the log posterior vanishes at $$\boldsymbol{\mu}_N$$, numerically with random matrices.)

## Specific cases

**1. The exercise itself: $$\boldsymbol{\Sigma}_c=\mathbf{I}$$, general Gaussian prior.** Put $$\boldsymbol{\Sigma}=\mathbf{I}$$:

$$\boldsymbol{\Sigma}_N=\left(\boldsymbol{\Sigma}_0^{-1}+N\mathbf{I}\right)^{-1},\qquad \boldsymbol{\mu}_N=\boldsymbol{\Sigma}_N\left(\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0+N\bar{\mathbf{x}}\right).$$

**2. $$\boldsymbol{\Sigma}_c=\mathbf{I}$$ and an isotropic prior $$\mathcal{N}(\boldsymbol{\mu}_0,\sigma^2\mathbf{I})$$.** Now every matrix is a multiple of $$\mathbf{I}$$, so the inverses are scalars:

$$\boldsymbol{\Sigma}_N=\frac{1}{1/\sigma^2+N}\mathbf{I}=\frac{\sigma^2}{1+N\sigma^2}\mathbf{I},\qquad
\boldsymbol{\mu}_N=\frac{\boldsymbol{\mu}_0/\sigma^2+N\bar{\mathbf{x}}}{1/\sigma^2+N}=\frac{1}{1+N\sigma^2}\boldsymbol{\mu}_0+\frac{N\sigma^2}{1+N\sigma^2}\bar{\mathbf{x}}.$$

The two weights are non-negative and sum to one.

**3. A zero-mean prior $$\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})$$.** Set $$\boldsymbol{\mu}_0=\mathbf{0}$$ in case 2:

$$\boldsymbol{\mu}_N=\frac{N\sigma^2}{1+N\sigma^2}\,\bar{\mathbf{x}},\qquad \boldsymbol{\Sigma}_N=\frac{\sigma^2}{1+N\sigma^2}\mathbf{I}.$$

The posterior mean is the sample mean shrunk towards the origin. This is the same shrinkage as in regularised (ridge) least squares.

**4. One dimension.** With $$\sigma_x^2$$ the known data variance and prior $$\mathcal{N}(\mu_0,\sigma_0^2)$$:

$$\sigma_N^2=\left(\frac1{\sigma_0^2}+\frac{N}{\sigma_x^2}\right)^{-1},\qquad \mu_N=\sigma_N^2\left(\frac{\mu_0}{\sigma_0^2}+\frac{N\bar{x}}{\sigma_x^2}\right).$$

**5. Limits, which also serve as checks on the algebra.**

| Situation | Effect on the result |
|---|---|
| $$N=0$$ (no data) | $$\boldsymbol{\mu}_N=\boldsymbol{\mu}_0$$, $$\boldsymbol{\Sigma}_N=\boldsymbol{\Sigma}_0$$: the posterior is the prior |
| $$N\to\infty$$ | $$\boldsymbol{\mu}_N\to\bar{\mathbf{x}}$$ and $$\boldsymbol{\Sigma}_N\approx\boldsymbol{\Sigma}/N\to\mathbf{0}$$: the data overwhelm the prior and the estimate is the maximum likelihood one |
| $$\sigma^2\to\infty$$ (vague prior), case 2 | $$\boldsymbol{\mu}_N\to\bar{\mathbf{x}}$$, $$\boldsymbol{\Sigma}_N\to\mathbf{I}/N$$: the posterior is the likelihood |
| $$\sigma^2\to 0$$ (very confident prior), case 2 | $$\boldsymbol{\mu}_N\to\boldsymbol{\mu}_0$$: the data are ignored |

**6. Which classes are uncertain.** The posterior covariance $$\boldsymbol{\Sigma}_N\approx\boldsymbol{\Sigma}_c/N_c$$ shrinks as $$1/N_c$$, so a class with few training objects has a wide posterior over its mean. This is the situation in which the book says the Bayesian treatment helps most (it is most useful when there is little data).

See also [Exercise 4.1](chapter4-exercise-4-1.html), which is the same calculation for linear regression weights, and the [Chapter 5 interactive for Figure 5.2](chapter5-class-conditional-gaussians.html) for the maximum likelihood fits per class.
