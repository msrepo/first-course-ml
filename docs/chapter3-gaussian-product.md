---
title: The product of two Gaussians
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML"></script>

# The product of two Gaussians

Chapter 3 notes. A Gaussian prior combined with a Gaussian likelihood gives a Gaussian posterior (Bayesian linear regression, Section 3.7). The reason is that multiplying two Gaussian densities gives something with a Gaussian shape. This note derives that result for two Gaussians with means $$\mu_0,\mu_1$$ and variances $$\sigma_0^2,\sigma_1^2$$. An [interactive page](chapter3-gaussian-product.html) lets you move the two means and variances and watch the product.

## Setup

In the book's notation a univariate Gaussian density in $$x$$ is

$$
p_0(x)=\mathcal{N}(\mu_0,\sigma_0^2)=\frac{1}{\sqrt{2\pi\sigma_0^2}}\exp\Big\{-\frac{1}{2\sigma_0^2}(x-\mu_0)^2\Big\},
$$

and $$p_1(x)=\mathcal{N}(\mu_1,\sigma_1^2)$$ is the same with $$\mu_1,\sigma_1^2$$. We multiply the two densities as functions of the same $$x$$. (Multiplying two Gaussian *random variables* is a different operation, and its result is not Gaussian.)

## The result

$$
p_0(x)\,p_1(x)=c\;\mathcal{N}(\mu,\sigma^2),
$$

with

$$
\sigma^2=\frac{\sigma_0^2\sigma_1^2}{\sigma_0^2+\sigma_1^2}\;\Big(\text{i.e. } \frac{1}{\sigma^2}=\frac{1}{\sigma_0^2}+\frac{1}{\sigma_1^2}\Big),
\qquad
\mu=\frac{\sigma_1^2\mu_0+\sigma_0^2\mu_1}{\sigma_0^2+\sigma_1^2},
$$

$$
c=\frac{1}{\sqrt{2\pi(\sigma_0^2+\sigma_1^2)}}\exp\Big\{-\frac{(\mu_0-\mu_1)^2}{2(\sigma_0^2+\sigma_1^2)}\Big\}.
$$

## Derivation

**Step 1: write the product with every constant.**

$$
p_0(x)p_1(x)=\frac{1}{2\pi\sigma_0\sigma_1}\exp\{-E\},
\qquad
E=\frac{(x-\mu_0)^2}{2\sigma_0^2}+\frac{(x-\mu_1)^2}{2\sigma_1^2}.
$$

**Step 2: clear the denominators.** Multiply $$E$$ by $$2\sigma_0^2\sigma_1^2$$:

$$
2\sigma_0^2\sigma_1^2E=\sigma_1^2(x-\mu_0)^2+\sigma_0^2(x-\mu_1)^2=a\,x^2-2b\,x+c_0,
$$

with

$$
a=\sigma_0^2+\sigma_1^2,\qquad
b=\sigma_1^2\mu_0+\sigma_0^2\mu_1,\qquad
c_0=\sigma_1^2\mu_0^2+\sigma_0^2\mu_1^2 .
$$

(The constant is called $$c_0$$ so that $$c$$ is free for the area below.)

**Step 3: complete the square.** For any quadratic, $$a x^2-2bx+c_0=a\,(x-b/a)^2+(c_0-b^2/a)$$. So the centre is

$$
\mu=\frac{b}{a}=\frac{\sigma_1^2\mu_0+\sigma_0^2\mu_1}{\sigma_0^2+\sigma_1^2}.
$$

For the leftover, put $$c_0-b^2/a$$ over the denominator $$a$$. The numerator is

$$
(\sigma_1^2\mu_0^2+\sigma_0^2\mu_1^2)(\sigma_0^2+\sigma_1^2)-(\sigma_1^2\mu_0+\sigma_0^2\mu_1)^2 .
$$

When this is expanded, the $$\sigma_1^4\mu_0^2$$ and $$\sigma_0^4\mu_1^2$$ terms cancel and what remains is $$\sigma_0^2\sigma_1^2(\mu_0^2-2\mu_0\mu_1+\mu_1^2)$$. Hence

$$
c_0-\frac{b^2}{a}=\frac{\sigma_0^2\sigma_1^2(\mu_0-\mu_1)^2}{\sigma_0^2+\sigma_1^2}.
$$

**Step 4: back to the exponent.** Divide by $$2\sigma_0^2\sigma_1^2$$:

$$
E=\frac{(x-\mu)^2}{2\sigma^2}+K,
\qquad
\sigma^2=\frac{\sigma_0^2\sigma_1^2}{\sigma_0^2+\sigma_1^2},
\quad
K=\frac{(\mu_0-\mu_1)^2}{2(\sigma_0^2+\sigma_1^2)} .
$$

The first term has the form of the exponent of a Gaussian with mean $$\mu$$ and variance $$\sigma^2$$. It is a genuine Gaussian because $$1/\sigma^2=1/\sigma_0^2+1/\sigma_1^2>0$$, so the quadratic opens downwards and the exponent decays at both ends. The second term does not contain $$x$$.

**Step 5: the constants.** The product is

$$
\frac{1}{2\pi\sigma_0\sigma_1}\,e^{-K}\exp\Big\{-\frac{(x-\mu)^2}{2\sigma^2}\Big\}.
$$

Multiply and divide by $$\sqrt{2\pi}\,\sigma$$ so that the last factor becomes the normalised density $$\mathcal{N}(\mu,\sigma^2)=\tfrac{1}{\sqrt{2\pi}\sigma}\exp\{-(x-\mu)^2/(2\sigma^2)\}$$:

$$
\frac{\sqrt{2\pi}\,\sigma}{2\pi\sigma_0\sigma_1}\,e^{-K}\;\mathcal{N}(\mu,\sigma^2).
$$

Now $$\sigma/(\sigma_0\sigma_1)=1/\sqrt{\sigma_0^2+\sigma_1^2}$$, so the constant in front is $$\dfrac{1}{\sqrt{2\pi(\sigma_0^2+\sigma_1^2)}}e^{-K}=c$$, as stated. The two normalising factors and $$e^{-K}$$ combine into a single Gaussian density in the gap $$\mu_0-\mu_1$$, with the two variances added.

**Step 6: $$c$$ is the area under the product.** Integrate Step 5's product over $$x$$. Since $$\int\exp\{-(x-\mu)^2/(2\sigma^2)\}dx=\sqrt{2\pi}\,\sigma$$,

$$
\int p_0(x)p_1(x)\,dx=\frac{1}{2\pi\sigma_0\sigma_1}e^{-K}\sqrt{2\pi}\,\sigma=c .
$$

The product of two densities is therefore not itself a density, because its area is $$c\ne 1$$. The factor $$\mathcal{N}(\mu,\sigma^2)$$ is the normalised part.

**Step 7: an independent check of $$c$$.** Let $$X\sim\mathcal{N}(\mu_0,\sigma_0^2)$$ and $$Y=X+\epsilon$$ with $$\epsilon\sim\mathcal{N}(0,\sigma_1^2)$$ independent of $$X$$. The density of $$Y$$ at $$\mu_1$$ is $$\int p_1\text{-style}\ \mathcal{N}(x,\sigma_1^2)\big|_{\mu_1}\,p_0(x)\,dx=\int p_0(x)p_1(x)\,dx$$. The moment-generating function of $$\mathcal{N}(\mu,\sigma^2)$$ is $$\exp\{\mu t+\sigma^2t^2/2\}$$, and for independent variables moment-generating functions multiply, so $$Y$$ has the MGF $$\exp\{\mu_0t+(\sigma_0^2+\sigma_1^2)t^2/2\}$$, that of $$\mathcal{N}(\mu_0,\sigma_0^2+\sigma_1^2)$$. Its density at $$\mu_1$$ is exactly $$c$$. This route never completes a square.

## What it means

- **Precisions add.** The precision is the inverse of the variance. The product is narrower than either input.
- **The mean is precision-weighted.** $$\mu$$ lies between $$\mu_0$$ and $$\mu_1$$, closer to the one with the smaller variance, the more certain one.
- **Equal variances:** the mean is $$(\mu_0+\mu_1)/2$$ and the variance is halved.
- **Wide prior or wide likelihood:** as $$\sigma_0^2\to\infty$$ the mean becomes $$\mu_1$$ (the data dominate), and as $$\sigma_1^2\to\infty$$ it becomes $$\mu_0$$ (the prior dominates).

**A hand check.** Take $$\mu_0=0,\ \sigma_0^2=1$$ and $$\mu_1=4,\ \sigma_1^2=4$$. Then $$a=5$$, $$b=4\cdot 0+1\cdot 4=4$$ and $$c_0=16$$. The centre is $$\mu=4/5=0.8$$, the leftover is $$c_0-b^2/a=16-16/5=12.8$$, which matches $$\sigma_0^2\sigma_1^2(\mu_0-\mu_1)^2/(\sigma_0^2+\sigma_1^2)=4\cdot16/5$$, and $$\sigma^2=4/5=0.8$$.

## Link to Bayesian inference

Read one Gaussian as a prior and the other, viewed as a function of the unknown, as a likelihood centred on a measured value $$\mu_1$$. The product is the unnormalised posterior: its precision is the prior precision plus the data precision, and its mean is the precision-weighted average of the prior mean and the data.

## The vector version

For $$D$$-dimensional Gaussians $$\mathcal{N}(\boldsymbol{\mu}_0,\boldsymbol{\Sigma}_0)$$ and $$\mathcal{N}(\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_1)$$, matching the quadratic and linear terms in $$\mathbf{x}$$ in the same way gives

$$
\boldsymbol{\Sigma}^{-1}=\boldsymbol{\Sigma}_0^{-1}+\boldsymbol{\Sigma}_1^{-1},
\qquad
\boldsymbol{\mu}=\boldsymbol{\Sigma}\big(\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0+\boldsymbol{\Sigma}_1^{-1}\boldsymbol{\mu}_1\big).
$$

In Bayesian linear regression the prior is $$p(\mathbf{w})=\mathcal{N}(\boldsymbol{\mu}_0,\boldsymbol{\Sigma}_0)$$. The likelihood $$\mathcal{N}(\mathbf{X}\mathbf{w},\sigma^2\mathbf{I})$$, viewed as a function of $$\mathbf{w}$$, is Gaussian in $$\mathbf{w}$$ with precision $$\mathbf{X}^T\mathbf{X}/\sigma^2$$ and centre $$(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{t}$$, the least-squares solution. The rule above then gives the posterior

$$
\boldsymbol{\Sigma}_w=\Big(\tfrac{1}{\sigma^2}\mathbf{X}^T\mathbf{X}+\boldsymbol{\Sigma}_0^{-1}\Big)^{-1},
\qquad
\boldsymbol{\mu}_w=\boldsymbol{\Sigma}_w\Big(\tfrac{1}{\sigma^2}\mathbf{X}^T\mathbf{t}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0\Big),
$$

the same result as the Chapter 3 derivation.
