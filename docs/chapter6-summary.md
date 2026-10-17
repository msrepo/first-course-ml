---
title: Chapter 6 summary, clustering
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML"></script>

# Chapter 6: Clustering, a quick recap

Short revision notes. Exercises are not covered.

## The problem

So far every task was **supervised**: each object $$\mathbf{x}_n$$ came with a target $$t_n$$. Now there are only the objects. Clustering groups them so that objects in the same group are similar and objects in different groups are not. What "similar" means is a modelling choice, for example squared distance for real-valued data. Typical uses: grouping customers (to base recommendations on similar customers) or genes with similar activity over time (to guess the function of an unknown gene from its known neighbours).

## K-means

A cluster is a representative point $$\boldsymbol{\mu}_k$$, the mean of the objects assigned to it. Let $$z_{nk}=1$$ if object $$n$$ is in cluster $$k$$ (each object in exactly one cluster):

$$
\boldsymbol{\mu}_k = \frac{\sum_n z_{nk}\mathbf{x}_n}{\sum_n z_{nk}} .
$$

The means need the assignments and the assignments need the means, so alternate:

1. Assign each object to its nearest mean, $$(\mathbf{x}_n-\boldsymbol{\mu}_k)^T(\mathbf{x}_n-\boldsymbol{\mu}_k)$$.
2. If no assignment changed, stop. Otherwise recompute each mean and go back to step 1.

The procedure converges to a **local** minimum of the total squared distance $$D=\sum_n\sum_k z_{nk}(\mathbf{x}_n-\boldsymbol{\mu}_k)^T(\mathbf{x}_n-\boldsymbol{\mu}_k)$$, not necessarily the global one. The result depends on the starting means, so run it from several random starts and keep the lowest $$D$$.

**Choosing $$K$$.** $$D$$ keeps falling as $$K$$ grows (it reaches 0 when every object is its own cluster), so it cannot choose $$K$$, the same failure as likelihood for model order in Chapter 2. There is no clean rule. Look at the purpose, for example pick the $$K$$ that gives the best recommendations or the best classification on validation data.

**Where it fails.** One point per cluster and squared distance mean clusters must be roughly round blobs of similar size: it cannot separate concentric rings (same centre), or clusters stretched so that some points are nearer the other cluster's centre.

### Kernel K-means

Use the kernel idea from Chapter 5. The distance to a cluster mean can be expanded so that the data appear only in inner products. Replace each inner product by a kernel $$k(\mathbf{x}_n,\mathbf{x}_m)$$ and the algorithm works in an implicit feature space, with no need to compute the means. Initialise by the assignments $$z_{nk}$$ (for example from ordinary K-means), then repeatedly compute kernelised distances and reassign. With a Gaussian kernel it separates the rings. It also lets you cluster any data type that has a kernel (text, graphs).

## Mixture models

K-means is too crude a notion of cluster. A mixture model describes each cluster as a probability density, so it can carry shape (spread and orientation).

**Generative story.** To produce one object: (1) pick component $$k$$ with probability $$\pi_k$$ ($$\sum_k\pi_k=1$$), (2) draw $$\mathbf{x}_n$$ from that component's density, e.g. $$\mathcal{N}(\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$$. The indicators $$z_{nk}$$ are never observed. They are **latent variables**.

**Likelihood.** Marginalising out the component choice:

$$
p(\mathbf{x}_n\mid\Lambda,\boldsymbol{\pi}) = \sum_{k=1}^{K}\pi_k\,p(\mathbf{x}_n\mid\Lambda_k),
\qquad
\mathcal{L} = \sum_n \log\sum_k \pi_k\,p(\mathbf{x}_n\mid\Lambda_k).
$$

The sum inside the log makes direct maximisation hard.

### The EM algorithm

Introduce a distribution $$q_{nk}$$ over components for each object ($$q_{nk}>0$$, $$\sum_kq_{nk}=1$$) and use **Jensen's inequality**, $$\log\mathbb{E}\{f\}\ge\mathbb{E}\{\log f\}$$, to get a lower bound $$\mathcal{B}$$ on $$\mathcal{L}$$. Maximising the bound (with Lagrange multipliers for the sum-to-one constraints) gives four updates, repeated until nothing changes:

$$
q_{nk} = \frac{\pi_k\,p(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{\sum_j \pi_j\,p(\mathbf{x}_n\mid\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}
\qquad\text{(E step)}
$$

$$
\pi_k=\frac1N\sum_n q_{nk},\qquad
\boldsymbol{\mu}_k=\frac{\sum_n q_{nk}\mathbf{x}_n}{\sum_n q_{nk}},\qquad
\boldsymbol{\Sigma}_k=\frac{\sum_n q_{nk}(\mathbf{x}_n-\boldsymbol{\mu}_k)(\mathbf{x}_n-\boldsymbol{\mu}_k)^T}{\sum_n q_{nk}}
\qquad\text{(M step)}
$$

**Intuition.**
- $$q_{nk}$$ is the posterior probability that object $$n$$ came from component $$k$$: prior $$\pi_k$$ times likelihood, normalised, the same form as the Bayes classifier of Chapter 5.
- $$\pi_k$$ is the average membership, $$\boldsymbol{\mu}_k$$ and $$\boldsymbol{\Sigma}_k$$ are membership-weighted mean and covariance.
- This is K-means with **soft** assignments (probabilities, not 0/1) and with covariances learned. Replacing $$q_{nk}$$ by $$z_{nk}$$ in the mean update gives exactly the K-means mean.
- The bound and the log-likelihood increase at every iteration.

**Practical points.**
- EM also only finds local maxima, and different starting values give different answers (relabelling components gives identical likelihood, so there are always several equivalent maxima). Run from many starts and compare likelihoods.
- Soft assignments keep information: an object with memberships 0.53 and 0.45 is nearly ambiguous, and hard assignment throws that away.
- On the data where K-means failed (stretched clusters), the mixture with $$K=2$$ finds the true structure.

### Choosing the number of components

The log-likelihood rises with $$K$$ (many small, tall densities fit the data points ever better), so it cannot choose $$K$$ on the training data. But unlike $$D$$ in K-means, the **likelihood on held-out data** works: use cross-validation and look for where it stops improving. It gives a rough range, not a sharp answer, which still beats K-means.

### Other component distributions

The component density can be anything.
- **Covariance restrictions** when data are scarce: diagonal (independent dimensions) or isotropic ($$\sigma_k^2\mathbf{I}$$). Only the $$\boldsymbol{\Sigma}_k$$ update changes.
- **Binary data:** a product of Bernoullis per component with probabilities $$p_{kd}$$. The E step uses the Bernoulli likelihood and the M step sets $$p_{kd}$$ to the membership-weighted average of dimension $$d$$.

### MAP versions and Bayesian mixtures

- **MAP with EM:** multiply the likelihood by a prior (e.g. beta on each $$p_{kd}$$) and maximise the posterior. The result is regularised, and a beta$$(1,1)$$ prior recovers maximum likelihood. It stops a probability from being pinned at exactly 0 or 1, which would veto a cluster for a new object.
- **Bayesian mixtures (MCMC):** sample cluster assignments and parameters, even the number of components. The output is a distribution over clusterings, which suits questions like "how often are objects $$n$$ and $$m$$ in the same cluster?", something EM cannot answer. A single clustering is hard to extract from it.

## Take-aways

1. Clustering is unsupervised: group similar objects, with "similar" being a modelling choice.
2. K-means alternates assignment and mean update, finds a local optimum, and suits round, similar-sized clusters. Kernels extend it.
3. Mixture models are a generative story with latent component labels. EM alternates soft assignment (E) and weighted parameter updates (M).
4. Neither the K-means distance nor the training likelihood can choose the number of clusters. Held-out likelihood or the downstream task can.
5. Both K-means and EM are local-optimum methods: restart from many random initialisations.
