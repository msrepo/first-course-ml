---
title: Exercise 4.2, the Laplace approximation to the coin posterior
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_SVG"></script>

# Exercise 4.2: the Laplace approximation to the coin posterior

Chapter 4 notes.

**The exercise.** In Chapter 3 the posterior over $$r$$, the probability that a coin lands heads, was computed with a beta prior and a binomial likelihood. The prior is

$$
p(r\mid\alpha,\beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\,r^{\alpha-1}(1-r)^{\beta-1},
$$

and the likelihood, for $$y$$ heads in $$N$$ throws, is

$$
p(y\mid r,N)=\binom{N}{y}r^{y}(1-r)^{N-y}.
$$

Compute the Laplace approximation to the posterior. The MAP value $$\hat r$$ comes out in closed form.

The Laplace approximation is $$\mathcal{N}(\mu,\sigma^2)$$ with $$\mu=\hat r=\arg\max_r\log g(r)$$ and $$\sigma^2=-1\big/\frac{\partial^2\log g}{\partial r^2}\big|_{r=\hat r}$$, where $$g(r)=p(y\mid r,N)\,p(r\mid\alpha,\beta)$$. With one variable, the Hessian is the second derivative.

## Step 1: the function g and its log

Multiply the likelihood by the prior and collect the powers of $$r$$ and $$1-r$$:

$$
g(r)=\binom{N}{y}\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\;r^{\alpha+y-1}(1-r)^{\beta+N-y-1}.
$$

Write $$\delta=\alpha+y$$ and $$\gamma=\beta+N-y$$ (the posterior parameters from Chapter 3, where the posterior is $$\mathrm{Beta}(\delta,\gamma)$$). The coefficient in front does not contain $$r$$, so

$$
\log g(r)=(\delta-1)\log r+(\gamma-1)\log(1-r)+\text{const}.
$$

## Step 2: the first derivative

Using $$\frac{d}{dr}\log r=\frac1r$$ and $$\frac{d}{dr}\log(1-r)=-\frac{1}{1-r}$$:

$$
\frac{\partial\log g}{\partial r}=\frac{\delta-1}{r}-\frac{\gamma-1}{1-r}.
$$

## Step 3: the mode

Set the derivative to zero and cross-multiply (both $$r$$ and $$1-r$$ are non-zero inside the interval):

$$
(\delta-1)(1-r)=(\gamma-1)\,r
\;\Longrightarrow\;
\delta-1=r\,(\delta+\gamma-2),
$$

so

$$
\boxed{\hat r=\frac{\delta-1}{\delta+\gamma-2}=\frac{\alpha+y-1}{\alpha+\beta+N-2}}
$$

This is the mode of a $$\mathrm{Beta}(\delta,\gamma)$$ density. It lies inside $$(0,1)$$ only if $$\delta>1$$ and $$\gamma>1$$. If either is at most 1, the density does not have an interior peak, and the Laplace recipe, which needs a peak with a negative second derivative, does not apply.

## Step 4: the second derivative

Use $$\frac{d}{dr}\frac1r=-\frac1{r^2}$$ and $$\frac{d}{dr}\frac{1}{1-r}=+\frac{1}{(1-r)^2}$$. The second term already carries a minus sign, so both pieces come out negative:

$$
\frac{\partial^2\log g}{\partial r^2}=-\frac{\delta-1}{r^2}-\frac{\gamma-1}{(1-r)^2}.
$$

For $$\delta>1$$ and $$\gamma>1$$ this is negative for every $$r$$ in $$(0,1)$$, so $$\hat r$$ is a maximum and the only stationary point.

## Step 5: evaluate at the mode

Write both $$\hat r$$ and $$1-\hat r$$ over the common denominator $$\delta+\gamma-2$$:

$$
\hat r=\frac{\delta-1}{\delta+\gamma-2},\qquad 1-\hat r=\frac{\gamma-1}{\delta+\gamma-2}.
$$

Each term then simplifies, with one factor cancelling:

$$
\frac{\delta-1}{\hat r^2}=\frac{(\delta+\gamma-2)^2}{\delta-1},
\qquad
\frac{\gamma-1}{(1-\hat r)^2}=\frac{(\delta+\gamma-2)^2}{\gamma-1}.
$$

Adding them over a common denominator, and using $$\frac{1}{\delta-1}+\frac{1}{\gamma-1}=\frac{\delta+\gamma-2}{(\delta-1)(\gamma-1)}$$,

$$
\frac{\partial^2\log g}{\partial r^2}\Big|_{\hat r}=-\frac{(\delta+\gamma-2)^3}{(\delta-1)(\gamma-1)}.
$$

## Step 6: the Laplace approximation

The variance is minus the inverse of that number:

$$
\sigma^2=\frac{(\delta-1)(\gamma-1)}{(\delta+\gamma-2)^3}=\frac{(\alpha+y-1)(\beta+N-y-1)}{(\alpha+\beta+N-2)^3}.
$$

Since $$\hat r(1-\hat r)=(\delta-1)(\gamma-1)/(\delta+\gamma-2)^2$$, this can also be written

$$
\sigma^2=\frac{\hat r(1-\hat r)}{\alpha+\beta+N-2}.
$$

So

$$
\boxed{\,p(r\mid y,N)\approx\mathcal{N}\Big(\hat r,\ \frac{\hat r(1-\hat r)}{\alpha+\beta+N-2}\Big)\,}
$$

## Checks

- **A flat prior** ($$\alpha=\beta=1$$): $$\hat r=y/N$$, the maximum-likelihood estimate, and $$\sigma^2=\hat r(1-\hat r)/N$$, the familiar variance of an estimated proportion.
- **A symmetric posterior** ($$\delta=\gamma$$): $$\hat r=\tfrac12$$ and $$\sigma^2=1/(8(\delta-1))$$, which shrinks as $$\delta$$ grows.
- **Against the exact posterior.** The true posterior is $$\mathrm{Beta}(\delta,\gamma)$$, with mean $$\delta/(\delta+\gamma)$$ and variance $$\delta\gamma/\big((\delta+\gamma)^2(\delta+\gamma+1)\big)$$. A hand example: $$\alpha=\beta=1$$, $$N=10$$, $$y=6$$, so $$\delta=7$$ and $$\gamma=5$$. The Laplace approximation is $$\mathcal{N}(0.6,\ 0.024)$$. The exact posterior has mean $$7/12\approx0.583$$ and variance $$35/1872\approx0.0187$$. The mode and the mean differ because the beta density is skewed, and the Gaussian is slightly wider than the true spread.
- **Large counts.** As $$\delta$$ and $$\gamma$$ grow, the mean and mode move together and the beta becomes nearly symmetric, so the approximation improves.
- **Edge effects.** The Gaussian lives on the whole real line, while $$r$$ lies in $$[0,1]$$. When $$\hat r$$ is near 0 or 1 with few counts, the Gaussian puts mass outside the interval, as in the gamma example.

## Why this is not exact

In Exercise 4.1 the log posterior was exactly quadratic, so the Laplace approximation was exact. Here $$\log g$$ contains $$\log r$$ and $$\log(1-r)$$, which are not quadratic, so the Taylor expansion about the mode drops higher-order terms and the approximation is only approximate. The binomial likelihood page shows the hump that this posterior is built from.
