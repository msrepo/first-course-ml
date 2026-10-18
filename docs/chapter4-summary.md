---
title: Chapter 4 summary, Bayesian inference
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML"></script>

# Chapter 4: Bayesian inference, a quick recap

Short revision notes. Exercises are not covered.

## The problem

Chapter 3's closed-form posteriors worked because the prior and likelihood were conjugate. Most useful models are not: for example, **binary classification** has no conjugate pair. We can still evaluate the numerator of Bayes' rule, but the normalising integral cannot be done. Chapter 4 gives three ways to cope, all illustrated on one classification problem.

## The running example: a binary response model

Each object has attributes $$\mathbf{x}_n$$ and a label $$t_n \in \{0,1\}$$.

- **Likelihood.** The labels are conditionally independent given $$\mathbf{w}$$. Each label is a coin toss whose probability of being 1 comes from passing a linear score through the **sigmoid**, which squashes any real number into $$(0,1)$$:

$$
P(T_n = 1 \mid \mathbf{x}_n,\mathbf{w}) = \frac{1}{1+\exp(-\mathbf{w}^T\mathbf{x}_n)} = P_n,
\qquad
p(\mathbf{t}\mid\mathbf{X},\mathbf{w}) = \prod_n P_n^{\,t_n}(1-P_n)^{1-t_n}.
$$

- **Prior.** $$\mathbf{w}\sim\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})$$.
- **Posterior.** $$p(\mathbf{w}\mid\mathbf{X},\mathbf{t},\sigma^2) = Z^{-1} g(\mathbf{w})$$ with $$g(\mathbf{w}) = p(\mathbf{t}\mid\mathbf{X},\mathbf{w})\,p(\mathbf{w}\mid\sigma^2)$$. We can compute $$g$$ at any $$\mathbf{w}$$ but not the constant $$Z^{-1}$$.
- **Prediction** needs an average over the posterior, $$P(t_{\text{new}}=1\mid\mathbf{x}_{\text{new}},\ldots)=\mathbb{E}_{p(\mathbf{w}\mid\ldots)}\{P(T_{\text{new}}=1\mid\mathbf{x}_{\text{new}},\mathbf{w})\}$$, which is also not available in closed form.

The three options: (1) find the single best $$\mathbf{w}$$, (2) approximate the posterior with a simpler density, (3) draw samples from the posterior.

## Option 1: the MAP estimate

Maximise $$\log g(\mathbf{w})$$ (prior times likelihood). The normaliser does not depend on $$\mathbf{w}$$, so it does not affect where the maximum is. There is no closed form, so use an iterative method, **Newton–Raphson**: repeatedly move to where the local tangent (or, for a maximum, the local quadratic model) crosses zero,

$$
\mathbf{w} \leftarrow \mathbf{w} - \mathbf{H}^{-1}\,\nabla \log g,
$$

where $$\mathbf{H}$$ is the Hessian. For this model

$$
\nabla \log g = -\tfrac{1}{\sigma^2}\mathbf{w} + \sum_n \mathbf{x}_n\,(t_n - P_n),
\qquad
\mathbf{H} = -\tfrac{1}{\sigma^2}\mathbf{I} - \sum_n \mathbf{x}_n\mathbf{x}_n^T P_n(1-P_n).
$$

$$\mathbf{H}$$ is negative definite for every $$\mathbf{w}$$ (because $$0<P_n<1$$), so there is a single maximum and the iteration finds it.

The decision boundary is where $$P=0.5$$, i.e. $$\hat{\mathbf{w}}^T\mathbf{x}=0$$, a straight line. The weakness: one value of $$\mathbf{w}$$ ignores the remaining uncertainty and gives **over-confident** predictions, even far from the data.

## Option 2: the Laplace approximation

Replace the posterior with a Gaussian that matches its peak and its curvature there. Expand $$\log g$$ to second order around the mode $$\hat{\mathbf{w}}$$ (the first-order term vanishes at a maximum):

$$
p(\mathbf{w}\mid\mathbf{X},\mathbf{t},\sigma^2) \approx \mathcal{N}(\hat{\mathbf{w}},\ \boldsymbol{\Sigma}),
\qquad
\boldsymbol{\Sigma} = -\mathbf{H}^{-1}\ \text{at}\ \hat{\mathbf{w}}.
$$

The mean is the MAP point and the covariance is the inverse curvature. We already have both from Newton–Raphson, so the approximation is nearly free.

- It is accurate near the mode and degrades away from it. It works well when the true posterior looks roughly Gaussian and poorly when it is skewed (the book checks this on a gamma density).
- The prediction integral is still not closed form, but sampling from a Gaussian is easy, so average the sigmoid over many samples:
  $$\frac{1}{N_s}\sum_s \frac{1}{1+\exp(-\mathbf{w}_s^T\mathbf{x}_{\text{new}})}$$
- Averaging over $$\mathbf{w}$$ "smudges" the decision boundary: predicted probabilities move toward 0.5 away from the data, which fixes much of the MAP over-confidence.

## Option 3: sampling, Metropolis–Hastings

Expectations can be estimated by averaging over samples, even if the density cannot be written down. (Darts example: the chance of hitting the target is just the fraction of throws that hit it.) Metropolis–Hastings produces samples from the posterior knowing only $$g$$:

1. Start anywhere.
2. **Propose** a move from the current sample, e.g. $$\mathbf{w}^*\sim\mathcal{N}(\mathbf{w}_{s-1},\boldsymbol{\Sigma})$$ (a random walk).
3. Compute the ratio $$r = g(\mathbf{w}^*)/g(\mathbf{w}_{s-1})$$. The normaliser cancels in the ratio. The symmetry of the Gaussian proposal means the proposal densities cancel too.
4. If $$r\ge 1$$ accept. Otherwise accept with probability $$r$$ (compare $$r$$ with a uniform draw). If rejected, repeat the old sample.

The chain moves uphill always and downhill sometimes, so it spends time in proportion to the posterior density.

- **Burn-in:** discard the early samples before the chain reaches the target distribution. Check convergence by running several chains from different starts and comparing them.
- **Using the samples:** marginals of a single parameter come from ignoring the others, and predictions come from averaging the sigmoid over the samples, as for Laplace.
- **Difficulties:** a posterior with several separate peaks (the chain may never cross between them), or strongly correlated parameters (a circular proposal in a long thin valley is mostly rejected). Better proposals and diagnostics exist but are not covered.

## Comparing the three

| | MAP | Laplace | Metropolis–Hastings |
|---|---|---|---|
| Result | One parameter vector | A Gaussian over $$\mathbf{w}$$ | Samples from the true posterior |
| Uncertainty in $$\mathbf{w}$$ | Ignored | Kept, but only as Gaussian | Kept in full |
| Cost | Low | Low (reuses the MAP work) | Highest |
| Main risk | Over-confident predictions | Poor if the posterior is not Gaussian-like | Slow mixing, convergence checks needed |

## Take-aways

1. Without conjugacy the posterior is known only up to a constant, $$g(\mathbf{w})$$.
2. The MAP estimate needs only $$g$$ and an optimiser, but throws away uncertainty.
3. Laplace turns the MAP point and curvature into a Gaussian posterior approximation.
4. MH samples from the true posterior using only ratios of $$g$$, at extra cost and with convergence care.
5. In every case, predictions average over parameter uncertainty (where it is kept), and the sampling-based average replaces integrals that cannot be done.

## An applied example: Laplace approximation for segmentation uncertainty

Zepf et al., ["Laplacian Segmentation Networks Improve Epistemic Uncertainty Quantification"](https://papers.miccai.org/miccai-2024/442-Paper1339.html) (MICCAI 2024; [arXiv](https://arxiv.org/abs/2303.13123)), apply the chapter's recipe to a medical image segmentation network. First the network is trained to a single best set of weights. This is a MAP-style solution, the point described in [Option 1: the MAP estimate](#option-1-the-map-estimate). Instead of stopping there, they treat that point as the centre of a Gaussian over the weights, using the curvature (Hessian) at the point to set its spread. That is exactly [Option 2: the Laplace approximation](#option-2-the-laplace-approximation).

The practical difficulty is scale. In this chapter the model has two weights, so the Hessian is a 2×2 matrix that can be formed and inverted. A segmentation network has millions of weights and a prediction for every pixel, so the full Hessian cannot be built. The authors use a much cheaper structured approximation of it, designed to cope with the network's skip connections, so the Laplace recipe still works at this size.

The approximate posterior is then used the way the chapter prescribes for predictions: draw several weight vectors from the Gaussian and look at all of the resulting predictions instead of a single one. Where the sampled networks disagree about a pixel, the model does not know the answer, which is *epistemic* uncertainty. Uncertainty that comes from the data itself (*aleatoric*) is modelled separately in the network's output. The disagreement is used to flag out-of-distribution images, where an ordinary network tends to stay confidently wrong. The authors report evaluating it on three public medical segmentation datasets, with better out-of-distribution detection than MC dropout and deep ensembles.

