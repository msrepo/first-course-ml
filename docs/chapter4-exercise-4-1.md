---
title: Exercise 4.1, the Laplace approximation is exact for a linear Gaussian model
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_SVG"></script>

# Exercise 4.1: the Laplace approximation is exact for a linear Gaussian model

Chapter 4 notes.

**The exercise.** A dataset has $$N$$ observations $$\mathbf{x}_n$$ (each $$D$$-dimensional) with real-valued targets $$t_n$$. The model is $$p(t_n\mid\mathbf{x}_n,\mathbf{w})=\mathcal{N}(\mathbf{w}^T\mathbf{x}_n,1)$$. Assume the observations are independent and put a Gaussian prior $$p(\mathbf{w})=\mathcal{N}(\boldsymbol{\mu}_0,\boldsymbol{\Sigma}_0)$$ on the weights. Show that the Laplace approximation to the posterior is equal to the true posterior.

## What has to be shown

The Laplace approximation (Equation 4.13) is $$\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$$ with

$$
\boldsymbol{\mu}=\hat{\mathbf{w}}=\arg\max_{\mathbf{w}}\log g(\mathbf{w}),
\qquad
\boldsymbol{\Sigma}=-\mathbf{H}^{-1},\quad \mathbf{H}=\frac{\partial^2\log g(\mathbf{w})}{\partial\mathbf{w}\,\partial\mathbf{w}^T}\Big|_{\mathbf{w}=\hat{\mathbf{w}}},
$$

where $$g(\mathbf{w};\mathbf{X},\mathbf{t})=p(\mathbf{t}\mid\mathbf{X},\mathbf{w})\,p(\mathbf{w})$$ is the prior times the likelihood, so that the posterior is $$Z^{-1}g$$. The true posterior is Gaussian, $$\mathcal{N}(\boldsymbol{\mu}_w,\boldsymbol{\Sigma}_w)$$ (Chapter 3). Two Gaussians are equal exactly when their means and covariances agree, so the proof has two parts:

1. $$\hat{\mathbf{w}}=\boldsymbol{\mu}_w$$,
2. $$-\mathbf{H}^{-1}=\boldsymbol{\Sigma}_w$$.

## Warm-up: the Laplace approximation of a univariate Gaussian

Let $$p(x)=\mathcal{N}(\mu,\sigma^2)$$. Then

$$
\log p(x)=-\frac{(x-\mu)^2}{2\sigma^2}-\tfrac12\log(2\pi\sigma^2).
$$

- First derivative: $$-(x-\mu)/\sigma^2$$, which is zero at $$x=\mu$$. The mode is the mean.
- Second derivative: $$-1/\sigma^2$$, the same for every $$x$$.
- Laplace variance: minus the inverse of the second derivative, which is $$\sigma^2$$.

So the Laplace approximation of $$\mathcal{N}(\mu,\sigma^2)$$ is $$\mathcal{N}(\mu,\sigma^2)$$ itself. The $$\pi$$ appears only in the normaliser, a constant in $$x$$ with zero derivative. This is the mechanism of the whole exercise: a Gaussian's log density is a quadratic, its curvature is constant, and the curvature fixes the covariance.

## Step 1: the likelihood and the log of g

With independent observations and unit noise variance, the likelihood is the product of the single-observation Gaussians:

$$
p(\mathbf{t}\mid\mathbf{X},\mathbf{w})=\prod_{n=1}^{N}\mathcal{N}(\mathbf{w}^T\mathbf{x}_n,1)=(2\pi)^{-N/2}\exp\Big\{-\tfrac12\sum_n(t_n-\mathbf{w}^T\mathbf{x}_n)^2\Big\}
=\mathcal{N}(\mathbf{X}\mathbf{w},\mathbf{I}),
$$

since the $$n$$th entry of $$\mathbf{X}\mathbf{w}$$ is $$\mathbf{w}^T\mathbf{x}_n$$. Taking logs of the likelihood times the prior and dropping everything without $$\mathbf{w}$$:

$$
\log g(\mathbf{w})=-\tfrac12(\mathbf{t}-\mathbf{X}\mathbf{w})^T(\mathbf{t}-\mathbf{X}\mathbf{w})-\tfrac12(\mathbf{w}-\boldsymbol{\mu}_0)^T\boldsymbol{\Sigma}_0^{-1}(\mathbf{w}-\boldsymbol{\mu}_0)+\text{const}.
$$

## Step 2: expand in w

Multiply out both quadratics. The terms without $$\mathbf{w}$$ go into the constant:

$$
\log g(\mathbf{w})=-\tfrac12\mathbf{w}^T\big(\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}\big)\mathbf{w}+\mathbf{w}^T\big(\mathbf{X}^T\mathbf{t}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0\big)+\text{const}.
$$

The quadratic part has $$\mathbf{X}^T\mathbf{X}$$ because $$\sum_n\mathbf{x}_n\mathbf{x}_n^T=\mathbf{X}^T\mathbf{X}$$, and the linear part has $$\mathbf{X}^T\mathbf{t}$$ because $$\sum_nt_n\mathbf{x}_n=\mathbf{X}^T\mathbf{t}$$.

## Step 3: gradient and Hessian

Two rules: for a symmetric matrix $$\mathbf{A}$$ and a vector $$\mathbf{a}$$,

$$
\frac{\partial}{\partial\mathbf{w}}\tfrac12\mathbf{w}^T\mathbf{A}\mathbf{w}=\mathbf{A}\mathbf{w},
\qquad
\frac{\partial}{\partial\mathbf{w}}\mathbf{a}^T\mathbf{w}=\mathbf{a}.
$$

Applying them, with $$\mathbf{A}=\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}$$ (symmetric, since both terms are):

$$
\frac{\partial\log g}{\partial\mathbf{w}}=-\big(\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}\big)\mathbf{w}+\mathbf{X}^T\mathbf{t}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0,
\qquad
\frac{\partial^2\log g}{\partial\mathbf{w}\,\partial\mathbf{w}^T}=-\big(\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}\big).
$$

The Hessian does not contain $$\mathbf{w}$$: the curvature is the same everywhere, not only at the mode. It is also negative definite, because $$\mathbf{X}^T\mathbf{X}$$ is positive semi-definite and $$\boldsymbol{\Sigma}_0^{-1}$$ is positive definite, so $$\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}$$ is positive definite and invertible. This holds even when $$\mathbf{X}^T\mathbf{X}$$ is singular (for example with more weights than data points).

## Step 4: the mode

Set the gradient to zero and solve:

$$
\big(\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}\big)\hat{\mathbf{w}}=\mathbf{X}^T\mathbf{t}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0
\;\Longrightarrow\;
\hat{\mathbf{w}}=\big(\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}\big)^{-1}\big(\mathbf{X}^T\mathbf{t}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0\big).
$$

Since the Hessian is negative definite, this is a maximum, and it is the only stationary point.

## Step 5: the Laplace approximation

$$
\boldsymbol{\Sigma}=-\mathbf{H}^{-1}=\big(\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}\big)^{-1},
\qquad
p(\mathbf{w}\mid\mathbf{X},\mathbf{t})\approx\mathcal{N}\Big(\hat{\mathbf{w}},\ \big(\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}\big)^{-1}\Big).
$$

## Step 6: the true posterior

This step finds the true posterior, independently of the Laplace calculation, so that Step 7 has something to compare with. The claim is that the posterior is Gaussian, and the step reads off its mean and covariance by matching coefficients.

**One variable first** (one weight $$w$$, one data point $$(x,t)$$, unit noise, prior $$\mathcal{N}(\mu_0,\sigma_0^2)$$). The log posterior is the log likelihood plus the log prior, plus a constant:

$$
-\tfrac12(t-wx)^2-\frac{(w-\mu_0)^2}{2\sigma_0^2}+\text{const}.
$$

Expand both squares and collect powers of $$w$$ (the terms without $$w$$ go into the constant):

$$
-\tfrac12\big(t^2-2txw+x^2w^2\big)-\frac{w^2-2\mu_0w+\mu_0^2}{2\sigma_0^2}
=-\tfrac12\,w^2\Big(x^2+\frac1{\sigma_0^2}\Big)+w\Big(xt+\frac{\mu_0}{\sigma_0^2}\Big)+\text{const}.
$$

Now write the log of a Gaussian $$\mathcal{N}(\mu_w,\sigma_w^2)$$ the same way:

$$
-\frac{(w-\mu_w)^2}{2\sigma_w^2}=-\tfrac12\,\frac{w^2}{\sigma_w^2}+w\,\frac{\mu_w}{\sigma_w^2}+\text{const}.
$$

If the two expressions are the same function of $$w$$ up to a constant, the coefficient of $$w^2$$ must agree and so must the coefficient of $$w$$:

$$
\frac{1}{\sigma_w^2}=x^2+\frac1{\sigma_0^2},
\qquad
\frac{\mu_w}{\sigma_w^2}=xt+\frac{\mu_0}{\sigma_0^2}.
$$

Matching coefficients is legitimate because the difference of two quadratics in $$w$$ that agree for every $$w$$ is a quadratic that is zero everywhere, and that is possible only if all its coefficients are zero.

**A number.** Take $$x=1$$, $$t=3$$, $$\mu_0=0$$ and $$\sigma_0^2=1$$. Then $$1/\sigma_w^2=1+1=2$$, so $$\sigma_w^2=0.5$$, and $$\mu_w=0.5\times(3+0)=1.5$$. The posterior mean sits halfway between the prior mean and the data value, because their precisions are equal.

**The matrix version.** The log posterior is the quadratic of Step 2:

$$
-\tfrac12\mathbf{w}^T\big(\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}\big)\mathbf{w}+\mathbf{w}^T\big(\mathbf{X}^T\mathbf{t}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0\big)+\text{const}.
$$

A Gaussian $$\mathcal{N}(\boldsymbol{\mu}_w,\boldsymbol{\Sigma}_w)$$ has log density, after expanding $$-\tfrac12(\mathbf{w}-\boldsymbol{\mu}_w)^T\boldsymbol{\Sigma}_w^{-1}(\mathbf{w}-\boldsymbol{\mu}_w)$$,

$$
-\tfrac12\mathbf{w}^T\boldsymbol{\Sigma}_w^{-1}\mathbf{w}+\mathbf{w}^T\boldsymbol{\Sigma}_w^{-1}\boldsymbol{\mu}_w+\text{const}.
$$

Matching the quadratic parts and the linear parts:

$$
\boldsymbol{\Sigma}_w^{-1}=\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1},
\qquad
\boldsymbol{\Sigma}_w^{-1}\boldsymbol{\mu}_w=\mathbf{X}^T\mathbf{t}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0,
$$

so $$\boldsymbol{\mu}_w=\boldsymbol{\Sigma}_w\big(\mathbf{X}^T\mathbf{t}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0\big)$$. This is the Chapter 3 result with noise variance 1. The role of $$1/\sigma_w^2$$ is played by $$\boldsymbol{\Sigma}_w^{-1}$$, the role of $$x^2$$ by $$\mathbf{X}^T\mathbf{X}$$, and the role of $$xt$$ by $$\mathbf{X}^T\mathbf{t}$$.

**How this relates to Steps 3 to 5.** The Laplace steps differentiate the same quadratic and solve for its peak. This step reads the coefficients off directly. Both extract the same information from one quadratic, which is why the two answers agree in Step 7.

## Step 7: comparison

- Step 4 gives $$\hat{\mathbf{w}}=(\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1})^{-1}(\mathbf{X}^T\mathbf{t}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0)=\boldsymbol{\mu}_w$$.
- Step 5 gives $$-\mathbf{H}^{-1}=(\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1})^{-1}=\boldsymbol{\Sigma}_w$$.

The Laplace approximation is $$\mathcal{N}(\boldsymbol{\mu}_w,\boldsymbol{\Sigma}_w)$$, exactly the true posterior. $$\blacksquare$$

## Why it had to work

The Laplace approximation is a second-order Taylor expansion of $$\log g$$ about the mode:

$$
\log g(\mathbf{w})\approx\log g(\hat{\mathbf{w}})+\tfrac12(\mathbf{w}-\hat{\mathbf{w}})^T\mathbf{H}(\mathbf{w}-\hat{\mathbf{w}}).
$$

Here $$\log g$$ is a quadratic in $$\mathbf{w}$$ (Step 2), and a quadratic has no derivatives of order three or higher, so the expansion drops nothing. This is why the Hessian in Step 3 does not depend on $$\mathbf{w}$$. In the binary response model of Chapter 4, by contrast, the Hessian contains $$P_n(1-P_n)$$ with $$P_n=1/(1+\exp(-\mathbf{w}^T\mathbf{x}_n))$$, which depends on $$\mathbf{w}$$, and the approximation is only approximate.

## The one-weight case, written out

Take $$D=1$$, targets $$t_n=wx_n+\text{noise}$$ with unit variance, and a prior $$\mathcal{N}(\mu_0,\sigma_0^2)$$. Then

$$
\log g(w)=-\tfrac12\sum_n(t_n-wx_n)^2-\frac{(w-\mu_0)^2}{2\sigma_0^2}+\text{const}.
$$

- First derivative: $$\sum_nx_n(t_n-wx_n)-(w-\mu_0)/\sigma_0^2$$.
- Second derivative: $$-\sum_nx_n^2-1/\sigma_0^2$$, with no $$w$$.
- Mode: setting the first derivative to zero gives $$w\big(\sum_nx_n^2+1/\sigma_0^2\big)=\sum_nx_nt_n+\mu_0/\sigma_0^2$$, so

$$
\hat{w}=\frac{\sigma_0^2\sum_nx_nt_n+\mu_0}{\sigma_0^2\sum_nx_n^2+1}.
$$

- Laplace variance: $$\dfrac{1}{\sum_nx_n^2+1/\sigma_0^2}=\dfrac{\sigma_0^2}{\sigma_0^2\sum_nx_n^2+1}$$.

These equal the exact posterior mean and variance: precision $$1/\sigma_0^2+\sum_nx_n^2$$ and mean $$\sigma_w^2\big(\mu_0/\sigma_0^2+\sum_nx_nt_n\big)$$.

## Checks

- **A very tight prior** ($$\sigma_0^2\to0$$): $$\hat{w}\to\mu_0$$ and the variance goes to 0. The prior wins.
- **A very wide prior** ($$\sigma_0^2\to\infty$$): $$\hat{w}\to\sum_nx_nt_n/\sum_nx_n^2$$, the least-squares slope through the origin, with variance $$1/\sum_nx_n^2$$. In the matrix case this is $$(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{t}$$.
- **One point** ($$N=1$$, $$x=1$$): $$\hat{w}=(\sigma_0^2t+\mu_0)/(\sigma_0^2+1)$$, a weighted average of the target and the prior mean.
- **Shapes** in the matrix version: the gradient is $$D\times1$$, the Hessian is $$D\times D$$, $$\mathbf{X}^T\mathbf{t}$$ is $$D\times1$$.
