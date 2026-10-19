---
title: Exercise 5.4, maximum likelihood for the multinomial class-conditional
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_SVG"></script>

# Exercise 5.4: maximum likelihood for the multinomial class-conditional

Chapter 5 notes.

**The exercise.** For a Bayesian classifier with multinomial class-conditionals, compute the maximum likelihood estimate of $$q_{cm}$$ for class $$c$$, given the $$N_c$$ $$M$$-dimensional objects $$\mathbf{x}_1,\dots,\mathbf{x}_{N_c}$$ of that class. (The exercise text is partly lost in the scanned copy; this is the reading consistent with Equation 5.7, which states the answer without derivation.)

**The model.** Object $$n$$ is a vector of word counts, $$x_{nm}$$ for word $$m$$, with total $$s_n=\sum_m x_{nm}$$. Given the class parameters $$\mathbf{q}_c=[q_{c1},\dots,q_{cM}]^T$$, with $$q_{cm}\ge0$$ and $$\sum_m q_{cm}=1$$ (Equation 5.6),

$$P(\mathbf{x}_n\mid\mathbf{q}_c)=\frac{s_n!}{\prod_{m}x_{nm}!}\prod_{m=1}^{M}q_{cm}^{x_{nm}}.$$

Drop the subscript $$c$$ while working: $$\mathbf{q}=\mathbf{q}_c$$ and $$N=N_c$$.

## Step 1: the log likelihood

The objects are independent given $$\mathbf{q}$$, so

$$\log P(\mathbf{X}\mid\mathbf{q})=\sum_{n=1}^{N}\left[\log s_n!-\sum_m\log x_{nm}!\right]+\sum_{m=1}^{M}\left(\sum_{n=1}^{N}x_{nm}\right)\log q_m.$$

The first sum does not contain $$\mathbf{q}$$, so it plays no part in the maximisation. Define the total count of word $$m$$ in the class,

$$X_m=\sum_{n=1}^{N}x_{nm},$$

so that, as a function of $$\mathbf{q}$$,

$$\log P(\mathbf{X}\mid\mathbf{q})=\sum_{m}X_m\log q_m+\text{const}.$$

The objects enter only through the word totals $$X_m$$ (the order of the objects, and which object a word came from, does not matter).

## Step 2: the constraint

We cannot just set the derivative to zero, because the $$q_m$$ must sum to one and a larger $$q_m$$ everywhere would always increase the likelihood. Use a Lagrange multiplier $$\lambda$$ for $$\sum_mq_m=1$$:

$$\mathcal{L}(\mathbf{q},\lambda)=\sum_mX_m\log q_m+\lambda\left(1-\sum_mq_m\right).$$

## Step 3: solve

Differentiate with respect to $$q_m$$ and set to zero:

$$\frac{\partial\mathcal{L}}{\partial q_m}=\frac{X_m}{q_m}-\lambda=0\quad\Longrightarrow\quad q_m=\frac{X_m}{\lambda}.$$

Substitute into the constraint to find $$\lambda$$:

$$\sum_mq_m=\frac1\lambda\sum_mX_m=1\quad\Longrightarrow\quad\lambda=\sum_{m}X_m=\sum_{n=1}^{N}s_n.$$

So $$\lambda$$ is the total number of words in the class. Hence

$$\boxed{\ \hat q_{cm}=\frac{\sum_{n=1}^{N_c}x_{nm}}{\sum_{n=1}^{N_c}\sum_{m'=1}^{M}x_{nm'}}\ }$$

This is the number of times word $$m$$ appears in class $$c$$ divided by the total number of words in class $$c$$: the observed proportion.

## Why this is a maximum

- $$\log q_m$$ is concave, so $$\sum_mX_m\log q_m$$ is a concave function of $$\mathbf{q}$$ (all $$X_m\ge0$$), and the constraint is linear. A stationary point of a concave function on that set is the global maximum.
- Check: the estimates sum to one, $$\sum_m\hat q_m=\sum_mX_m/\lambda=1$$.
- Numerical check: for random multinomial counts, perturbing $$\hat{\mathbf{q}}$$ in 20,000 random directions (renormalised) never gave a larger log likelihood.

## Remarks

**The answer in the book.** Equation 5.7 prints the answer with extra factorial terms. They are the constants of Step 1 that do not depend on $$\mathbf{q}$$ and cancel out of the maximisation; the ratio of counts above is the result.

**Zero counts.** If word $$m$$ never appears in class $$c$$, then $$X_m=0$$ and $$\hat q_{cm}=0$$. Then any new document containing that word has probability exactly zero under class $$c$$, whatever its other words say. This is the problem addressed by smoothing in Section 5.2.1.7, and it motivates the Dirichlet prior of the next exercise: a prior with parameter $$\alpha$$ gives a posterior (and MAP estimate) based on $$X_m+\alpha-1$$ and never reaches zero.

**Comparison with the Gaussian case.** In Exercise 5.3 the maximum likelihood fit of a class is also computed from only that class's objects: the sample mean and covariance (Equations 5.4 and 5.5), found by setting derivatives to zero. Here the extra step is the constraint on $$\mathbf{q}$$, handled by a Lagrange multiplier.
