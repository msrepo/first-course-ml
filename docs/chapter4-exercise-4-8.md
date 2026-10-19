---
title: Exercises 4.7 and 4.8, Laplace and Metropolis-Hastings for Poisson regression
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_SVG"></script>

# Exercises 4.7 and 4.8: Poisson regression, Laplace and Metropolis–Hastings

Chapter 4 notes. Both exercises use the model of [Exercise 4.6](chapter4-exercise-4-6.html). The sampler can be run in the browser: [Interactive: the Metropolis–Hastings random walk](chapter4-poisson-mcmc.html). The code is [scripts/ex4_8_poisson_mh.py](https://github.com/msrepo/first-course-ml/blob/main/scripts/ex4_8_poisson_mh.py).

## The model, from Exercise 4.6

With $$V_n = \exp(\mathbf{w}^T\mathbf{x}_n)$$ and a prior $$\mathbf{w}\sim\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})$$, the unnormalised posterior is $$g(\mathbf{w};\mathbf{X},\mathbf{t},\sigma^2)=p(\mathbf{t}\mid\mathbf{X},\mathbf{w})\,p(\mathbf{w})$$ and

$$\log g=\sum_{n=1}^N\left[t_n\mathbf{w}^T\mathbf{x}_n-\exp(\mathbf{w}^T\mathbf{x}_n)\right]-\frac{\mathbf{w}^T\mathbf{w}}{2\sigma^2}+\text{const}.$$

$$\frac{\partial\log g}{\partial\mathbf{w}}=\sum_n\mathbf{x}_n(t_n-V_n)-\frac{\mathbf{w}}{\sigma^2},\qquad
\frac{\partial^2\log g}{\partial\mathbf{w}\,\partial\mathbf{w}^T}=-\sum_nV_n\mathbf{x}_n\mathbf{x}_n^T-\frac{1}{\sigma^2}\mathbf{I}.$$

The Hessian is negative definite for every $$\mathbf{w}$$, so $$\log g$$ is concave and has a single maximum $$\hat{\mathbf{w}}$$, found by Newton–Raphson.

## Exercise 4.7: the Laplace approximation

Expand $$\log g$$ to second order about $$\hat{\mathbf{w}}$$. The first-order term vanishes because the gradient is zero there, so

$$\log g(\mathbf{w})\approx\log g(\hat{\mathbf{w}})-\tfrac12(\mathbf{w}-\hat{\mathbf{w}})^T\left(-\mathbf{H}\right)(\mathbf{w}-\hat{\mathbf{w}}),\qquad \mathbf{H}=\left.\frac{\partial^2\log g}{\partial\mathbf{w}\,\partial\mathbf{w}^T}\right\vert_{\hat{\mathbf{w}}}.$$

This is the exponent of a Gaussian, so

$$p(\mathbf{w}\mid\mathbf{X},\mathbf{t},\sigma^2)\approx\mathcal{N}(\hat{\mathbf{w}},\boldsymbol{\Sigma}),\qquad \boldsymbol{\Sigma}=\left(\sum_n\hat V_n\mathbf{x}_n\mathbf{x}_n^T+\frac{1}{\sigma^2}\mathbf{I}\right)^{-1},\quad \hat V_n=\exp(\hat{\mathbf{w}}^T\mathbf{x}_n).$$

Unit check: each term of $$\boldsymbol{\Sigma}^{-1}$$ is a precision, so $$\boldsymbol{\Sigma}$$ has the units of $$\mathbf{w}^2$$ as it should.

## Exercise 4.8: Metropolis–Hastings

**The idea.** We cannot sample from $$p(\mathbf{w}\mid\mathbf{X},\mathbf{t},\sigma^2)=g(\mathbf{w})/Z$$ directly, and we do not know $$Z$$. A random walk avoids needing it.

1. Start at some $$\mathbf{w}^{(0)}$$.
2. Propose $$\mathbf{w}^*\sim\mathcal{N}(\mathbf{w}^{(s)},\,\delta^2\mathbf{I})$$.
3. Accept with probability $$\min\left(1,\ g(\mathbf{w}^*)/g(\mathbf{w}^{(s)})\right)$$. The proposal is symmetric, so no proposal densities appear in the ratio, and $$Z$$ cancels. In logs: accept if $$\log u<\log g(\mathbf{w}^*)-\log g(\mathbf{w}^{(s)})$$ with $$u\sim U(0,1)$$.
4. If accepted, $$\mathbf{w}^{(s+1)}=\mathbf{w}^*$$. If rejected, $$\mathbf{w}^{(s+1)}=\mathbf{w}^{(s)}$$. The repeated sample is not an error: it is what makes the density of the samples correct.

Discard an initial burn-in, then histogram the rest.

**The step size.** If $$\delta$$ is too small almost every step is accepted but the walk crawls, so successive samples are strongly correlated. If it is too large almost every step is rejected. A common rule of thumb is $$\delta\approx 2.4/\sqrt{D}$$ times the posterior standard deviation, with $$D=2$$ weights here, which gives an acceptance rate of roughly 30%. The Laplace standard deviations give a convenient estimate of that scale.

## The comparison

Synthetic data: $$N=20$$ points, $$x_n=(1,u_n)$$ with $$u_n\sim U(-1.5,1.5)$$, true weights $$(1.0,\,0.7)$$, $$\sigma^2=10$$, 50,000 steps, 1,000 burn-in, step $$\delta=0.254$$ (acceptance rate 31%).

| | $$w_1$$ | $$w_2$$ |
|---|---|---|
| Laplace mean (the mode) | 1.0112 | 0.7810 |
| MCMC mean | 0.9899 | 0.7905 |
| Laplace sd | 0.1468 | 0.1522 |
| MCMC sd | 0.1473 | 0.1526 |
| Correlation, Laplace / MCMC | −0.555 / −0.557 | |

The two agree closely. The remaining difference is the skewness of the true posterior, which a Gaussian cannot represent: the MCMC mean sits slightly away from the mode (−0.021 for $$w_1$$) and the sample skewness of $$w_1$$ is −0.21. This gap shrinks as $$N$$ grows and the posterior becomes more Gaussian: the difference in the mean of $$w_1$$ is −0.0485 for $$N=6$$, −0.0212 for $$N=20$$ and −0.0073 for $$N=60$$.

**Which to use.** Laplace needs only a Newton–Raphson run and one matrix inverse, and is accurate when the posterior is close to Gaussian (many data points). Metropolis–Hastings is slower and needs tuning, but it converges to the true posterior whatever its shape, including skewness and heavy tails.

Use the [interactive](chapter4-poisson-mcmc.html) to watch the walk, the trace plots and the histograms against the Laplace Gaussian, and to vary $$N$$, $$\sigma^2$$, the step size and the burn-in.
