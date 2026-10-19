---
title: Chapter 4 summary, Bayesian inference
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_SVG"></script>

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

## Option 1: the MAP estimate {#option-1-map}

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

## Option 2: the Laplace approximation {#option-2-laplace}

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

## Option 3: sampling, Metropolis–Hastings {#option-3-mh}

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

## Applied examples from the literature

Short descriptions of papers that use the chapter's ideas on real data. Each paragraph links back to the option it illustrates.

### MAP: a classical refinement stage ([Option 1](#option-1-map))

["Two Projections Suffice for Cerebral Vascular Reconstruction"](https://papers.miccai.org/miccai-2024/806-Paper1481.html) (MICCAI 2024) rebuilds a 3D vessel volume from only two X-ray-style views. A U-Net first gives a rough 3D estimate. Then a MAP stage takes over: the volume is adjusted iteratively until its re-projections match the two views, while a prior that favours connected vessels keeps the answer plausible. This is the chapter's recipe in miniature: a data-fit term (the likelihood) plus a prior that regularises, and the single best volume is the one that maximises their product, just as MAP trades the likelihood against the prior in Chapter 3. The scans come from public brain MRA data, but the two views are simulated projections, and the authors note there is no validation on real fluoroscopy.

### MAP, then a Gaussian around it ([Option 1](#option-1-map) then [Option 2](#option-2-laplace))

Wang, Wells, Golland and Zhang, ["Efficient Laplace Approximation for Bayesian Registration Uncertainty Quantification"](https://pmc.ncbi.nlm.nih.gov/articles/PMC6533616/) (MICCAI 2018), is the closest match to the chapter's structure, and it does not use deep learning. A registration warps one brain scan onto another. The method first finds the MAP deformation by minimising the negative log posterior (a matching term plus a smoothness prior). It then approximates the posterior as a Gaussian centred on that deformation, with covariance equal to the inverse Hessian. The difficulty is size: a deformation of a 3D image has far too many parameters to form the Hessian directly, so the authors work in a low-dimensional frequency representation of the deformation, where the Hessian can be handled efficiently using products with vectors. The result is a map of how uncertain the registration is at each location, tested on real OASIS brain MRI.

### Laplace for neural networks, and keeping the Hessian small ([Option 2](#option-2-laplace))

Zepf et al., ["Laplacian Segmentation Networks Improve Epistemic Uncertainty Quantification"](https://papers.miccai.org/miccai-2024/442-Paper1339.html) (MICCAI 2024; [arXiv](https://arxiv.org/abs/2303.13123)), apply the recipe to a medical image segmentation network. First the network is trained to a single best set of weights, the MAP-style solution of [Option 1](#option-1-map). Then they treat that point as the centre of a Gaussian over the weights, using the curvature (Hessian) at the point to set its spread, which is [Option 2](#option-2-laplace). In this chapter the model has two weights, so the Hessian is a 2×2 matrix that can be formed and inverted. A segmentation network has millions of weights and a prediction for every pixel, so the authors use a much cheaper structured approximation of the Hessian, designed to cope with the network's skip connections. The approximate posterior is then used as the chapter prescribes for predictions: draw several weight vectors from the Gaussian and look at all of the resulting predictions. Where the sampled networks disagree about a pixel, the model does not know the answer, which is *epistemic* uncertainty. Uncertainty from the data itself (*aleatoric*) is modelled separately in the network's output. The disagreement flags out-of-distribution images, where an ordinary network tends to stay confidently wrong. The authors report evaluating it on three public medical segmentation datasets, with better out-of-distribution detection than MC dropout and deep ensembles.

["Last Layer Laplacian Pseudocoresets for Robust Medical Image Analysis"](https://papers.miccai.org/miccai-2025/0477-Paper2092.html) (MICCAI 2025) shrinks the problem differently: apply the Laplace approximation to the final layer only. After the network is trained, a Gaussian is placed over the last layer's weights, which are few enough for the Hessian to be handled easily, the same trick as the two-weight example in this chapter but for one layer of a network. The paper then fits a small set of pseudo-examples (a "pseudocoreset") against this posterior, aiming at robust, uncertainty-aware predictions under class imbalance without retraining the network. It is evaluated on skin-image datasets (ISIC 2019 and ASAN). A reviewer on the paper's page notes that last-layer Laplace is not new in itself and that the novelty is the pseudocoreset step.

### Metropolis–Hastings for a posterior over shapes ([Option 3](#option-3-mh))

Madsen et al., ["A Closest Point Proposal for MCMC-based Probabilistic Surface Registration"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620273.pdf) (ECCV 2020), is not a medical imaging paper, but it is motivated by medicine and uses CT-derived bone shapes. It treats registering one 3D surface onto another as inference: a Gaussian-process shape model is the prior, a point-distance model is the likelihood, and the resulting posterior over registrations can be evaluated up to a constant but not normalised, exactly the situation of this chapter. They sample it with Metropolis–Hastings. The standard random-walk proposal needs very long runs in high dimensions, so the paper designs an informed proposal that uses the geometry (a point on the template moves toward its closest point on the target) while staying a valid, stochastic proposal so the acceptance ratio remains well defined. The samples give an uncertainty for each point of the surface, and for completing partly missing shapes a set of posterior samples covers the ground truth better than the single most likely (MAP) shape. Data: 50 healthy femur meshes extracted from CT, and 10 face scans.
