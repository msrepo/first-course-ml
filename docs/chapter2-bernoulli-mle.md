---
title: Maximum likelihood for the Bernoulli parameter
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML"></script>

# Maximum likelihood for the Bernoulli parameter

Chapter 2 notes.

## 1. The Bernoulli distribution

A Bernoulli random variable describes one yes/no event. It takes the value $$x = 1$$ ("yes") or $$x = 0$$ ("no"), and has one parameter $$q$$ with $$0 \le q \le 1$$, the chance of a 1:

$$
P(X = 1) = q, \qquad P(X = 0) = 1 - q.
$$

Both cases fit in one formula, valid for $$x \in \{0, 1\}$$:

$$
P(X = x \mid q) = q^{x}\,(1 - q)^{1 - x}.
$$

Repeated draws are independent: seeing a 1 now says nothing about the next draw.

**Bernoulli versus binomial.** The binomial distribution counts the number of 1s, $$y$$, in $$n$$ independent draws:

$$
P(Y = y \mid q) = \binom{n}{y} q^{y} (1 - q)^{n - y}.
$$

The factor $$\binom{n}{y}$$ counts the orderings that give the same total. We are modelling one particular sequence $$x_1, \dots, x_N$$, so that factor does not appear. It would not change the optimum anyway, because it does not contain $$q$$.

## 2. Likelihood and log-likelihood

The draws are independent, so the probability of the whole sequence is a product:

$$
L = P(x_1, \dots, x_N \mid q) = \prod_{n=1}^{N} q^{x_n}(1 - q)^{1 - x_n}.
$$

Take the log, using $$\log(ab) = \log a + \log b$$ and $$\log(a^c) = c\log a$$:

$$
\log L = \sum_{n=1}^{N} \Big[ x_n \log q + (1 - x_n)\log(1 - q) \Big].
$$

## 3. First derivative

Two derivatives are needed:

- $$\dfrac{d}{dq}\log q = \dfrac{1}{q}$$
- $$\dfrac{d}{dq}\log(1 - q) = -\dfrac{1}{1 - q}$$ (chain rule: the inside, $$1 - q$$, has derivative $$-1$$)

Here $$x_n$$ is a data value, so it is a constant:

$$
\frac{\partial \log L}{\partial q} = \sum_{n=1}^{N} \left[ \frac{x_n}{q} - \frac{1 - x_n}{1 - q} \right].
$$

The terms depend on $$n$$ only through whether $$x_n$$ is 1 or 0, so only the number of 1s in the data matters. Write $$N_1 = \sum_n x_n$$. The number of 0s is then $$N - N_1$$, and

$$
\frac{\partial \log L}{\partial q} = \frac{N_1}{q} - \frac{N - N_1}{1 - q}.
$$

## 4. Solving for the maximum

Set the derivative to zero:

$$
\frac{N_1}{q} = \frac{N - N_1}{1 - q}
\;\Longrightarrow\;
N_1(1 - q) = q\,(N - N_1)
\;\Longrightarrow\;
N_1 = qN.
$$

$$
\boxed{\;\hat q = \frac{N_1}{N}\;}
$$

The maximum-likelihood estimate is the fraction of 1s in the data. It lies between 0 and 1 for any $$N_1$$ from 0 to $$N$$.

## 5. Checking that it is a maximum

Differentiate again. The derivative of $$1/q$$ is $$-1/q^2$$, and for $$1/(1-q)$$ the chain rule gives $$+1/(1-q)^2$$, which the minus sign in front turns into a minus:

$$
\frac{\partial^2 \log L}{\partial q^2} = -\frac{N_1}{q^2} - \frac{N - N_1}{(1 - q)^2}.
$$

$$N_1$$ and $$N - N_1$$ are non-negative counts, and $$q^2$$ and $$(1-q)^2$$ are positive, so each term is $$\le 0$$. The second derivative is negative for every $$q$$ in $$(0, 1)$$. Negative curvature means the log-likelihood bends downward like a hill, so the stationary point $$\hat q$$ is a maximum, and there is no other stationary point on $$(0,1)$$.

## 6. A picture to hold on to

For a dataset with one 1 and one 0, $$\log L = \log q + \log(1 - q)$$. Its graph is a symmetric hill with its top at $$q = 1/2$$, falling steeply toward $$-\infty$$ at both ends: a value of $$q$$ near 0 makes the observed 1 almost impossible, and a value near 1 does the same for the observed 0. Its slope, $$(1 - 2q)/\big(q(1-q)\big)$$, is positive to the left of the top, zero at the top and negative to the right. Setting the slope to zero finds the top.

## 7. Fisher information for the Bernoulli parameter

The Fisher information of a parameter is the expected curvature of the log-likelihood, with a minus sign:

$$
I(\theta) = -\,\mathbb{E}\!\left[\frac{\partial^2 \log L}{\partial \theta^2}\right].
$$

The Gaussian result $$\tfrac{1}{\sigma^2}\mathbf{X}^T\mathbf{X}$$ is this quantity for the weights of the linear Gaussian model. It is not the general definition, so the Bernoulli case does not need a $$\sigma^2$$. Here the parameter is the single number $$q$$, so the "matrix" is $$1\times 1$$.

**Two roles of $$q$$.** In maximum likelihood the data are fixed and $$q$$ is varied to find the peak. In Fisher information $$q$$ is held fixed and the data vary: the question is how sharply $$\log L$$ would bend around its peak for typical datasets that came from that value of $$q$$. The curvature depends on which dataset is drawn, so we average over datasets, and that average is the expectation. The true $$q$$ is not known in practice, so in use one plugs in $$\hat q$$.

**Derivation.** Start from the second derivative found in section 5:

$$
\frac{\partial^2 \log L}{\partial q^2} = -\frac{N_1}{q^2} - \frac{N - N_1}{(1 - q)^2}.
$$

At a fixed $$q$$ the only random quantity is $$N_1 = \sum_n x_n$$. Each $$x_n$$ has mean $$\mathbb{E}[x_n] = 1\cdot q + 0\cdot(1 - q) = q$$, so by linearity

$$
\mathbb{E}[N_1] = Nq, \qquad \mathbb{E}[N - N_1] = N(1 - q).
$$

Substituting, with the leading minus sign cancelling the minus signs:

$$
I(q) = \frac{Nq}{q^2} + \frac{N(1 - q)}{(1 - q)^2} = \frac{N}{q} + \frac{N}{1 - q} = \frac{N}{q(1 - q)},
$$

using $$q + (1 - q) = 1$$ in the numerator after putting the two fractions over $$q(1-q)$$.

$$
\boxed{\;I(q) = \frac{N}{q\,(1 - q)}\;}
$$

**Checks.**

- It is positive for $$0 < q < 1$$.
- It grows without bound as $$q \to 0$$ or $$q \to 1$$, matching the steep walls of the log-likelihood hill: near the ends the curve bends sharply, so the data pin $$q$$ down tightly.
- It is smallest at $$q = 1/2$$, where $$q(1 - q)$$ is largest. A fair coin is the hardest case to pin down.
- It is $$N$$ times the information of a single draw, because the log-likelihoods of independent draws add.

**Variance of the estimator.** $$N_1$$ is a sum of $$N$$ independent Bernoulli values, each with variance $$q(1 - q)$$, so $$\operatorname{Var}(N_1/N) = q(1 - q)/N = 1/I(q)$$. This is the same pattern as in the Gaussian model, where the inverse of $$\tfrac{1}{\sigma^2}\mathbf{X}^T\mathbf{X}$$ is the covariance of $$\hat{\mathbf{w}}$$. The estimator $$\hat q = N_1/N$$ reaches the Cramér–Rao lower bound: no unbiased estimator has a smaller variance.
