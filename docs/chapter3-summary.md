---
title: Chapter 3 summary, the Bayesian approach
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML"></script>

# Chapter 3: The Bayesian approach, a quick recap

Short revision notes. Exercises are not covered.

## The big idea

Chapter 2 showed that estimates of the parameters carry uncertainty. Chapter 3 takes the next step: treat the parameters themselves as **random variables**, put a distribution on them before seeing data (the **prior**), and update it with the data to get the **posterior**. Instead of one best value you keep a whole distribution that says which values are plausible and how sure we are.

## Bayes' rule for a parameter

$$
p(\theta \mid \text{data}) = \frac{p(\text{data} \mid \theta)\; p(\theta)}{p(\text{data})}
$$

- **Likelihood** $$p(\text{data}\mid\theta)$$: how well each value of $$\theta$$ explains the data. As a function of $$\theta$$ it is *not* a density, and it does not integrate to 1. More data makes it narrower.
- **Prior** $$p(\theta)$$: beliefs about $$\theta$$ before seeing data.
- **Marginal likelihood** $$p(\text{data})=\int p(\text{data}\mid\theta)p(\theta)\,d\theta$$: the likelihood averaged over all parameter values. It only normalises the posterior, but it is also useful for model comparison (below).
- **Posterior**: prior and likelihood combined. It shows what is still uncertain after seeing the data.

Only the shape matters when you work out the posterior: posterior $$\propto$$ likelihood $$\times$$ prior.

## Example 1: the coin game (beta prior, binomial likelihood)

Maximum likelihood from one short run of tosses can give an extreme value, and it hides how little data there is. The Bayesian version asks for a distribution over the head probability $$r$$.

- **Beta prior** $$\mathrm{Beta}(\alpha,\beta)$$ lives on $$[0,1]$$ and is flexible: flat, peaked, skewed, depending on $$\alpha,\beta$$. Mean $$\alpha/(\alpha+\beta)$$.
- **Conjugacy.** Beta prior with a binomial likelihood gives a beta posterior: with $$y$$ heads in $$N$$ tosses it is $$\mathrm{Beta}(\alpha + y,\; \beta + N - y)$$. The normalising constant is already known, so $$p(\text{data})$$ is not needed.
- **Pseudo-counts.** Read $$\alpha$$ and $$\beta$$ as imagined earlier heads and tails. A large $$\alpha+\beta$$ is a strong prior.
- **Common conjugate pairs:** beta with binomial, Gaussian with Gaussian, gamma with Gaussian (for the precision), Dirichlet with multinomial.

**Three priors.** Uniform ("know nothing"), a sharp peak at 0.5 ("fair coin") and a skew to heads ("biased"):

- Each toss shifts the posterior and, in general, narrows it, so uncertainty falls as data arrives.
- A strong prior barely moves for a modest amount of data. A weak prior moves quickly.
- With enough data the posteriors from different priors converge: the likelihood is a product over observations and swamps the single prior term.

**Predictions.** Predict by averaging over the posterior, $$\mathbb{E}_{p(r\mid y)}\{P(\text{future outcome}\mid r)\}$$, rather than plugging in one point estimate. The two agree more closely as the posterior gets narrower. With few data, the full average accounts for the remaining uncertainty and the point estimate ignores it.

## Marginal likelihood for choosing priors

Integrate $$r$$ out given the prior parameters: $$p(y\mid\alpha,\beta)=\int p(y\mid r)\,p(r\mid\alpha,\beta)\,dr$$. A higher value means the prior setting agrees better with the data, so it can pick among the three priors. For the beta-binomial case it has a closed form. Searching over $$\alpha,\beta$$ to maximise it is called **Type II maximum likelihood**. Caution: the prior then no longer represents beliefs held before seeing the data.

## Hyper-parameters and graphical models

- $$\alpha,\beta$$ can be random too, with a prior controlled by a **hyper-parameter**. Layers can be stacked, but the integrals soon become intractable (Chapter 4's approximations).
- **Graphical models** draw the structure: nodes are random variables, shaded nodes are observed, arrows are dependencies and a **plate** marks something repeated $$N$$ times.

## Example 2: Bayesian linear regression (Olympics data)

Model: $$\mathbf t = \mathbf X\mathbf w + \boldsymbol\epsilon$$, noise $$\epsilon_n\sim\mathcal N(0,\sigma^2)$$, with $$\sigma^2$$ taken as known.

- **Likelihood:** $$\mathcal N(\mathbf X\mathbf w,\ \sigma^2\mathbf I)$$.
- **Prior (conjugate):** $$\mathbf w\sim\mathcal N(\boldsymbol\mu_0,\boldsymbol\Sigma_0)$$. (The reason a Gaussian prior with a Gaussian likelihood gives a Gaussian posterior is worked out in [the product of two Gaussians](chapter3-gaussian-product.html).)
- **Posterior is Gaussian**, $$\mathcal N(\boldsymbol\mu_w,\boldsymbol\Sigma_w)$$, found by collecting the quadratic and linear terms in $$\mathbf w$$:

$$
\boldsymbol\Sigma_w = \Big(\tfrac{1}{\sigma^2}\mathbf X^T\mathbf X + \boldsymbol\Sigma_0^{-1}\Big)^{-1},
\qquad
\boldsymbol\mu_w = \boldsymbol\Sigma_w\Big(\tfrac{1}{\sigma^2}\mathbf X^T\mathbf t + \boldsymbol\Sigma_0^{-1}\boldsymbol\mu_0\Big).
$$

- **MAP and regularisation.** The posterior mean is the most probable $$\mathbf w$$ (the maximum a posteriori estimate). With a Gaussian likelihood it is the same as regularised least squares from Chapter 1; the prior plays the role of the penalty.
- **How the posterior evolves.** With a prior that treats $$w_0$$ and $$w_1$$ as independent, early data pin down the intercept first, and the posterior then tilts, showing that the data tie the two parameters together. As data accumulates, it contracts. Samples of $$\mathbf w$$ give lines that fan out under the prior and tighten around the trend under the posterior. A smaller noise $$\sigma^2$$ contracts it more.

**Prediction.** Average the model over the posterior:

$$
p(t_{\text{new}}\mid \mathbf x_{\text{new}},\mathbf X,\mathbf t,\sigma^2)=\mathcal N\!\big(\mathbf x_{\text{new}}^T\boldsymbol\mu_w,\ \ \sigma^2+\mathbf x_{\text{new}}^T\boldsymbol\Sigma_w\mathbf x_{\text{new}}\big).
$$

It looks like the maximum-likelihood predictive density, but it averages over every parameter setting that fits the data and prior, rather than committing to one.

## Marginal likelihood for choosing model order

The marginal likelihood of the data under a polynomial model is again Gaussian, $$\mathcal N\big(\mathbf X\boldsymbol\mu_0,\ \sigma^2\mathbf I+\mathbf X\boldsymbol\Sigma_0\mathbf X^T\big)$$, evaluated at the observed $$\mathbf t$$.

- On data generated from a cubic, it peaks sharply at the right order, using all the data and without refitting on several splits (unlike cross-validation).
- It depends on the prior. A tighter prior (small prior variance) penalises large coefficients, which can make higher-order models score better than the true one. "The model" means the order **and** the prior together.
- In general it is hard to compute, so cross-validation is often easier.

## Take-aways

1. Treat parameters as random variables; prior $$\times$$ likelihood gives the posterior.
2. Conjugate pairs give closed-form posteriors: beta with binomial, Gaussian with Gaussian.
3. Strong priors resist data; with enough data the prior stops mattering.
4. Predict by averaging over the posterior, not by plugging in a point.
5. The marginal likelihood compares priors and models, but is rarely tractable. Chapter 4 covers approximations.
