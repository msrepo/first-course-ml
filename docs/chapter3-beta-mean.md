---
title: Mean and variance of a beta random variable
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_SVG"></script>

# Mean and variance of a beta random variable

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

## Variance

Use $$\operatorname{var}\{r\} = \mathbb{E}[r^2] - (\mathbb{E}[r])^2$$. The mean is done, so only $$\mathbb{E}[r^2]$$ is new.

**Finding $$\mathbb{E}[r^2]$$.** The same steps as for the mean, with one more power of $$r$$:

$$
\mathbb{E}[r^2] = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \int_0^1 r^{\alpha+1}(1-r)^{\beta-1}\,dr .
$$

Matching $$r^{\alpha+1}=r^{a-1}$$ gives $$a=\alpha+2$$, with $$b=\beta$$. The identity gives the integral as $$\Gamma(\alpha+2)\Gamma(\beta)/\Gamma(\alpha+\beta+2)$$. The gamma arguments are now shifted by 2, so the recursion is applied twice to each:

$$
\Gamma(\alpha+2) = (\alpha+1)\,\alpha\,\Gamma(\alpha), \qquad
\Gamma(\alpha+\beta+2) = (\alpha+\beta+1)(\alpha+\beta)\,\Gamma(\alpha+\beta).
$$

The remaining gamma functions cancel, leaving

$$
\mathbb{E}[r^2] = \frac{\alpha(\alpha+1)}{(\alpha+\beta)(\alpha+\beta+1)} .
$$

**Subtracting the squared mean.**

$$
\operatorname{var}\{r\} = \frac{\alpha(\alpha+1)}{(\alpha+\beta)(\alpha+\beta+1)} - \frac{\alpha^2}{(\alpha+\beta)^2}.
$$

Put both fractions over $$(\alpha+\beta)^2(\alpha+\beta+1)$$. The numerator is

$$
\alpha(\alpha+1)(\alpha+\beta) - \alpha^2(\alpha+\beta+1)
= (\alpha^3+\alpha^2\beta+\alpha^2+\alpha\beta) - (\alpha^3+\alpha^2\beta+\alpha^2)
= \alpha\beta,
$$

so

$$
\boxed{\;\operatorname{var}\{r\} = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}\;}
$$

### Checks and special cases

- **Uniform ($$\alpha=\beta=1$$).** The formula gives $$1/12$$. The beta density with these parameters is the uniform density on $$[0,1]$$, with $$\mathbb{E}[r]=1/2$$ and $$\mathbb{E}[r^2]=\int_0^1 r^2\,dr = 1/3$$, so the variance is $$1/3-1/4=1/12$$.
- **Symmetric in $$\alpha$$ and $$\beta$$.** Swapping heads and tails should not change the spread, and it does not.
- **Equal parameters.** With $$\beta=\alpha$$ it reduces to $$1/[4(2\alpha+1)]$$, a function of $$\alpha$$ alone. The two parameters collapse to one because the density is symmetric about $$1/2$$: the mean is fixed, and $$\alpha=\beta$$ only controls how concentrated the peak is.
- **More imagined tosses, less spread.** The denominator grows like $$(\alpha+\beta)^3$$ and the numerator at most like $$(\alpha+\beta)^2$$, so the variance falls as $$\alpha+\beta$$ grows. A strong prior is a narrow density.
- **Positive** for all $$\alpha,\beta>0$$.

![Beta densities with the same mean, narrowing as alpha plus beta grows](figures/beta-variance-spread.svg)
