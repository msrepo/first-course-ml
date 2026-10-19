---
title: The product of two Gaussians
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML"></script>

# The product of two Gaussians

Chapter 3 notes. The Gaussian prior with a Gaussian likelihood in Bayesian linear regression works because multiplying two Gaussian densities gives something with a Gaussian shape. An [interactive page](chapter3-gaussian-product.html) lets you move the two means and variances and watch the product.

## The result

Multiply the two densities as functions of the same variable $$x$$ (this is different from multiplying two Gaussian random variables, whose product is not Gaussian):

$$
\mathcal{N}(x\mid\mu_0,\sigma_0^2)\,\mathcal{N}(x\mid\mu_1,\sigma_1^2) \;=\; c\;\mathcal{N}(x\mid\mu,\sigma^2),
$$

with

$$
\frac{1}{\sigma^2}=\frac{1}{\sigma_0^2}+\frac{1}{\sigma_1^2},
\qquad
\mu=\sigma^2\Big(\frac{\mu_0}{\sigma_0^2}+\frac{\mu_1}{\sigma_1^2}\Big)=\frac{\mu_0\sigma_1^2+\mu_1\sigma_0^2}{\sigma_0^2+\sigma_1^2},
\qquad
c=\mathcal{N}(\mu_0\mid\mu_1,\ \sigma_0^2+\sigma_1^2).
$$

## Why: complete the square

Take the log of the product and keep the terms that contain $$x$$:

$$
-\tfrac12\Big[\frac{(x-\mu_0)^2}{\sigma_0^2}+\frac{(x-\mu_1)^2}{\sigma_1^2}\Big]
= -\tfrac12\Big[x^2\Big(\frac1{\sigma_0^2}+\frac1{\sigma_1^2}\Big)-2x\Big(\frac{\mu_0}{\sigma_0^2}+\frac{\mu_1}{\sigma_1^2}\Big)\Big]+\text{const}.
$$

The log of $$\mathcal{N}(\mu,\sigma^2)$$ is $$-\tfrac12 x^2/\sigma^2 + x\mu/\sigma^2+\text{const}$$. Matching the coefficient of $$x^2$$ gives $$1/\sigma^2$$, and matching the coefficient of $$x$$ gives $$\mu/\sigma^2$$. That is the variance and the mean above. The constant $$c$$ is whatever is left over, and it equals the area under the product (the integral over $$x$$). It is large when the two means are close and shrinks as they move apart.

No normalising constant had to be computed along the way. Once the shape is recognised as Gaussian, its normaliser is known. This is the same trick as recognising a beta posterior in Chapter 3.

## What it means

- **Precisions add.** Precision is the inverse of the variance. The product is narrower than both inputs.
- **The mean is precision-weighted.** It lies between $$\mu_0$$ and $$\mu_1$$, closer to the one with the smaller variance, the more certain one.
- **Equal variances:** the mean is $$(\mu_0+\mu_1)/2$$ and the variance is halved.

**A hand check.** Take $$\mu_0=0,\ \sigma_0^2=1$$ and $$\mu_1=4,\ \sigma_1^2=4$$. The precisions are $$1$$ and $$0.25$$, which add to $$1.25$$, so $$\sigma^2=0.8$$. The mean is $$0.8\,(0/1+4/4)=0.8$$.

## Link to Bayesian inference

Read one Gaussian as a prior $$\mathcal{N}(\mu_0,\sigma_0^2)$$ and the other as a likelihood centred on a measured value $$\mu_1$$. The product is the unnormalised posterior: its precision is the prior precision plus the data precision, and its mean is the precision-weighted average of the prior mean and the data.

## The vector version

For $$D$$-dimensional Gaussians the same matching of the quadratic and linear terms gives

$$
\boldsymbol{\Sigma}^{-1}=\boldsymbol{\Sigma}_0^{-1}+\boldsymbol{\Sigma}_1^{-1},
\qquad
\boldsymbol{\mu}=\boldsymbol{\Sigma}\big(\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0+\boldsymbol{\Sigma}_1^{-1}\boldsymbol{\mu}_1\big).
$$

In Bayesian linear regression the prior is $$\mathcal{N}(\boldsymbol{\mu}_0,\boldsymbol{\Sigma}_0)$$ over $$\mathbf{w}$$. The likelihood $$\mathcal{N}(\mathbf{X}\mathbf{w},\sigma^2\mathbf{I})$$, viewed as a function of $$\mathbf{w}$$, is Gaussian in $$\mathbf{w}$$ with precision $$\mathbf{X}^T\mathbf{X}/\sigma^2$$ and "mean" $$(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{t}$$, the least-squares solution. Applying the rule above gives the posterior

$$
\boldsymbol{\Sigma}_w=\Big(\tfrac{1}{\sigma^2}\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}\Big)^{-1},
\qquad
\boldsymbol{\mu}_w=\boldsymbol{\Sigma}_w\Big(\tfrac{1}{\sigma^2}\mathbf{X}^T\mathbf{t}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0\Big),
$$

the same result as the Chapter 3 derivation.

## Checks

- The product is always narrower than either input, because the precisions add.
- With a very wide prior ($$\sigma_0^2\to\infty$$), the mean becomes $$\mu_1$$: the data dominate.
- With a very wide likelihood ($$\sigma_1^2\to\infty$$), the mean becomes $$\mu_0$$: the prior dominates.
