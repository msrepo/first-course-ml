---
title: The numerator of the posterior and what is intractable
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML"></script>

# The numerator of the posterior, and what is actually intractable

Chapter 4 notes, for the binary response model: labels $$t_n\in\{0,1\}$$, weights $$\mathbf{w}$$, sigmoid likelihood and a Gaussian prior. An [interactive page](chapter4-evaluate-numerator.html) works through every step below with sliders for $$\mathbf{w}$$ and $$\sigma^2$$.

## Bayes' rule for the weights

$$
p(\mathbf{w}\mid\mathbf{X},\mathbf{t},\sigma^2) = \frac{p(\mathbf{t}\mid\mathbf{X},\mathbf{w})\,p(\mathbf{w}\mid\sigma^2)}{p(\mathbf{t}\mid\mathbf{X},\sigma^2)}
= \frac{g(\mathbf{w})}{Z},
\qquad
Z=\int g(\mathbf{w})\,d\mathbf{w}.
$$

The numerator is $$g(\mathbf{w}) = p(\mathbf{t}\mid\mathbf{X},\mathbf{w})\,p(\mathbf{w}\mid\sigma^2)$$ and the denominator $$Z$$ is the marginal likelihood. The book writes the factor that multiplies $$g$$ as $$Z^{-1}$$: $$Z$$ is the normaliser (the integral of $$g$$), and the posterior divides by it. This is just a naming convention. Both $$Z$$ and $$1/Z$$ are equally unknown here.

## Evaluating the numerator at one w

Pick a specific weight vector. Two numbers multiply:

1. **The prior** (two weights, so $$D=2$$). It depends only on $$\mathbf{w}$$ and $$\sigma^2$$, not on the data:

$$
p(\mathbf{w}\mid\sigma^2)=(2\pi\sigma^2)^{-1}\exp\!\Big(-\frac{\mathbf{w}^T\mathbf{w}}{2\sigma^2}\Big).
$$

2. **The likelihood.** For each data point compute $$z_n=\mathbf{w}^T\mathbf{x}_n$$ and $$P_n=1/(1+e^{-z_n})$$, then take the probability the model gives to the **observed** label, $$P_n$$ if $$t_n=1$$ and $$1-P_n$$ if $$t_n=0$$. Multiply over all $$n$$:

$$
p(\mathbf{t}\mid\mathbf{X},\mathbf{w}) = \prod_n P_n^{\,t_n}(1-P_n)^{1-t_n}.
$$

Multiply the two to get $$g(\mathbf{w})$$. A product of many numbers below 1 becomes tiny, so in practice add logs: $$\log g = \log\text{likelihood}+\log\text{prior}$$.

A weight vector whose boundary separates the classes correctly gives each point a high probability for its own label, so the likelihood is near 1. A weight vector that points the wrong way gives most points a low probability, and the likelihood collapses. The prior then penalises large weights.

## The numerator is not the intractable part

$$g(\mathbf{w})$$ is a completely known function: give it any $$\mathbf{w}$$ and it returns a number. Evaluated for **every** $$\mathbf{w}$$ it is the shape of the posterior, only on the wrong vertical scale. With two weights you can evaluate it on a grid of $$(w_1,w_2)$$ and draw a heat map or contours.

What is intractable is turning that function into what Bayesian inference needs. Each of these is an integral over $$\mathbf{w}$$:

- the normaliser $$Z=\int g(\mathbf{w})\,d\mathbf{w}$$,
- the posterior mean and covariance, $$\mathbb{E}[\mathbf{w}]=\tfrac1Z\int\mathbf{w}\,g(\mathbf{w})\,d\mathbf{w}$$,
- the marginal of one weight, $$p(w_1\mid\text{data})=\tfrac1Z\int g(w_1,w_2)\,dw_2$$,
- predictions, $$P(t_{\text{new}}=1\mid\mathbf{x}_{\text{new}},\text{data})=\tfrac1Z\int\sigma(\mathbf{w}^T\mathbf{x}_{\text{new}})\,g(\mathbf{w})\,d\mathbf{w}$$,
- a way to draw samples from $$g/Z$$.

None has a closed form here, because the product of sigmoid factors with a Gaussian is not a recognisable density, so there are no formulas for its integrals or moments.

**The conjugate contrast.** In the coin example $$g$$ was proportional to a beta density. Recognising the family gave the mean, the variance and a sampler for free. Knowing $$g$$ pointwise was not what made that case tractable. Belonging to a family with known integrals was.

**Why a grid is not the answer in general.** With two weights, summing $$g$$ over a grid gives rough values for $$Z$$ and for the posterior mean. With $$D$$ weights, the number of grid points needed grows like (points per axis)$$^D$$, which quickly becomes impossible.

## How the three methods use only pointwise g

- **MAP:** maximise $$g$$ (the unknown $$Z$$ does not depend on $$\mathbf{w}$$, so it does not move the maximum).
- **Laplace:** fit a Gaussian using the location, height and curvature of $$g$$ at its peak.
- **Metropolis–Hastings:** compare $$g$$ at two points. The unknown $$Z$$ cancels in the ratio, and the samples then estimate every integral above by averaging.

The heat map of $$g$$ on the interactive page shows why the peak and the mean differ: the surface is skewed, so its centre of mass is not at its peak, and a symmetric Gaussian at the peak (Laplace) cannot capture that.

## The decision boundary in terms of the weights

The boundary is where $$P(t=1)=0.5$$, i.e. $$\mathbf{w}^T\mathbf{x}=0$$:

$$
w_1x_1+w_2x_2=0
\;\Longrightarrow\;
x_2=-\frac{w_1}{w_2}\,x_1 .
$$

- **Slope** $$=-w_1/w_2$$ and **intercept** $$=0$$: with no constant term, the boundary always passes through the origin.
- $$\mathbf{w}$$ is **perpendicular** to the boundary and points toward the class-1 side.
- Multiplying $$\mathbf{w}$$ by a positive constant leaves the line unchanged. It only changes how quickly the probability goes from 0 to 1 across it, which is why $$\sigma^2$$ affects the confidence of a prior draw but not its direction.
- A non-zero intercept needs a constant input: with $$\mathbf{x}=(1,x_1,x_2)$$ and weights $$(w_0,w_1,w_2)$$ the boundary is $$x_2=-\tfrac{w_1}{w_2}x_1-\tfrac{w_0}{w_2}$$.
