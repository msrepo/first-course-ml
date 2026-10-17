---
title: Expected value of a beta random variable
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML"></script>

# Expected value of a beta random variable

Chapter 3 notes. The beta density, for $$0 < r < 1$$ and $$\alpha,\beta>0$$:

$$
p(r) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\; r^{\alpha-1}(1-r)^{\beta-1}.
$$

## Two tools

- The gamma recursion $$\Gamma(n+1) = n\,\Gamma(n)$$. Note that $$\Gamma(n+1)$$ is $$n$$ times $$\Gamma(n)$$, not $$n+1$$.
- The normalising identity, which says the density integrates to 1:

$$
\int_0^1 r^{a-1}(1-r)^{b-1}\,dr = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}.
$$

For general exponents this integral has no simple antiderivative to evaluate between 0 and 1. The way to use the identity is to recognise the integrand as the right form and match $$a$$ and $$b$$.

## Derivation

Start from the definition of the expected value:

$$
\mathbb{E}[r] = \int_0^1 r\, p(r)\, dr = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \int_0^1 r^{\alpha}(1-r)^{\beta-1}\, dr .
$$

The constant leaves the integral. The extra factor $$r$$ merges with $$r^{\alpha-1}$$ to give $$r^{\alpha}$$. Matching $$r^{\alpha} = r^{a-1}$$ gives $$a = \alpha+1$$, and $$b=\beta$$ is unchanged. The identity gives the integral as $$\Gamma(\alpha+1)\Gamma(\beta)/\Gamma(\alpha+\beta+1)$$:

$$
\mathbb{E}[r] = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\cdot\frac{\Gamma(\alpha+1)\,\Gamma(\beta)}{\Gamma(\alpha+\beta+1)} .
$$

Apply the recursion to the two "+1" gamma functions, once each:

$$
\Gamma(\alpha+1) = \alpha\,\Gamma(\alpha), \qquad \Gamma(\alpha+\beta+1) = (\alpha+\beta)\,\Gamma(\alpha+\beta).
$$

Substituting, the gamma functions $$\Gamma(\alpha)$$, $$\Gamma(\beta)$$ and $$\Gamma(\alpha+\beta)$$ cancel between top and bottom and leave only the extra factors:

$$
\boxed{\;\mathbb{E}[r] = \frac{\alpha}{\alpha+\beta}\;}
$$

A common slip is to keep "$$\alpha+1$$" in the answer. The recursion turns $$\Gamma(\alpha+1)$$ into the factor $$\alpha$$, and $$\Gamma(\alpha+\beta+1)$$ into the factor $$\alpha+\beta$$.

## Special cases

![Four beta densities with the mean marked](figures/beta-special-cases.svg)

- **$$\alpha=\beta=1$$ (flat).** Every value of $$r$$ is equally likely, and the formula gives $$1/2$$. Direct check: $$\int_0^1 r\,dr = 1/2$$.
- **$$\alpha=2,\ \beta=1$$ ($$p(r)=2r$$).** The density leans to large $$r$$. The formula gives $$2/3$$. Direct check: $$\int_0^1 r\cdot 2r\,dr = 2/3$$.
- **$$\alpha=3,\ \beta=1$$ ($$p(r)=3r^2$$).** Even more weight on large $$r$$. The formula gives $$3/4$$. Direct check: $$\int_0^1 r\cdot 3r^2\,dr = 3/4$$.
- **$$\alpha=\beta$$ in general.** The density is symmetric about $$1/2$$, so the mean is $$1/2$$ for every $$\alpha$$. Larger $$\alpha=\beta$$ narrows the peak but does not move it.

## Other sanity checks

- The mean always lies between 0 and 1.
- It rises when $$\alpha$$ increases and falls when $$\beta$$ increases.
- **Pseudo-counts:** reading $$\alpha$$ as imagined heads and $$\beta$$ as imagined tails, the mean is the fraction of the $$\alpha+\beta$$ imagined tosses that were heads.
