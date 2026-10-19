# A First Course in Machine Learning

Notes by chapter (work in progress).

## Chapter 1: Linear modelling, a least-squares approach

**Fitting a line**

- [Interactive: fit a line to the Olympic data](chapter1-linear-modelling.html)

**Mathematical background**

- [Notes: vector and matrix notation, and differentiating with it](chapter1-vector-matrix-notation.html)

## Chapter 2: Linear modelling from a probabilistic view

**The noise model and the likelihood**

- [Interactive: likelihood of a Gaussian linear model](chapter2-gaussian-likelihood.html)

**The shape of the likelihood**

- [Interactive: the quadratic bowl (log-likelihood surface)](chapter2-quadratic-bowl.html)
- [Interactive: curvature of the likelihood (the Hessian)](chapter2-hessian.html)

**Model complexity and uncertainty**

- [Interactive: maximum likelihood favours complex models](chapter2-model-complexity.html)
- [Interactive: predictive error bars](chapter2-predictive-error-bars.html)

**Another example: the Bernoulli parameter**

- [Notes: maximum likelihood for the Bernoulli parameter](chapter2-bernoulli-mle.html)

## Chapter 3: The Bayesian approach

**Overview**

- [Notes: Chapter 3 summary](chapter3-summary.html)

**The coin example: beta prior, binomial likelihood**

- [Interactive: the beta density](chapter3-beta-density.html)
- [Interactive: the binomial likelihood](chapter3-binomial-likelihood.html)
- [Notes: posterior with a flat prior](chapter3-posterior-flat-prior.html)
- [Notes: mean and variance of a beta random variable](chapter3-beta-mean.html)

**Gaussian prior and Gaussian likelihood**

- [Notes: the product of two Gaussians, derivation](chapter3-gaussian-product.html)
- [Interactive: the product of two Gaussians](chapter3-gaussian-product-interactive.html)

## Chapter 4: Bayesian inference

**Overview**

- [Notes: Chapter 4 summary](chapter4-summary.html)

**The running example: the binary response model**

- [Interactive: the binary response dataset, with prior draws](chapter4-binary-data.html)

**Prior, likelihood and the unnormalised posterior**

- [Interactive: the likelihood read two ways (density of the data, likelihood of the weight)](chapter4-likelihood-two-views.html)
- [Interactive: evaluating the numerator of Bayes' rule at one w](chapter4-evaluate-numerator.html)
- [Notes: the numerator of the posterior and what is intractable](chapter4-numerator-and-posterior.html)

**MAP and Newton–Raphson**

- [Interactive: Newton–Raphson step by step](chapter4-newton-raphson.html)

**Laplace approximation**

- [Interactive: the gamma density and its Laplace approximation](chapter4-gamma-laplace.html)
- [Interactive: Laplace approximation for the binary response model](chapter4-laplace-binary.html)

**Sampling: Metropolis–Hastings**

- [Interactive: the random walk, trace plots and histograms against the Laplace approximation](chapter4-poisson-mcmc.html)
- [Interactive: when the random walk struggles (bimodal, skewed, curved, correlated targets)](chapter4-mcmc-failure-modes.html)

**Exercises**

- [Notes: Exercise 4.1, the Laplace approximation is exact for a linear Gaussian model](chapter4-exercise-4-1.html)
- [Notes: Exercise 4.2, the Laplace approximation to the coin posterior](chapter4-exercise-4-2.html)
- [Notes: Exercise 4.6, Poisson regression and the MAP solution](chapter4-exercise-4-6.html)
- [Notes: Exercises 4.7 and 4.8, Laplace and Metropolis–Hastings for Poisson regression](chapter4-exercise-4-8.html)
- [Interactive: Exercise 4.3, the coin posterior and its Laplace approximation](chapter4-beta-laplace.html)

## Chapter 5: Classification

**Overview**

- [Notes: Chapter 5 summary](chapter5-summary.html)

**Gaussian class-conditional densities**

- [Interactive: Figure 5.2, a Gaussian fitted to each class (Equations 5.4 and 5.5)](chapter5-class-conditional-gaussians.html)

**Exercises**

- [Notes: Exercise 5.1, the posterior of a class mean with a Gaussian prior](chapter5-exercise-5-1.html)
- [Notes: Exercise 5.4, maximum likelihood for the multinomial class-conditional](chapter5-exercise-5-4.html)

## Chapter 6: Clustering

**Overview**

- [Notes: Chapter 6 summary](chapter6-summary.html)

## Chapter 7: Principal components analysis and latent variable models

**Overview**

- [Notes: Chapter 7 summary](chapter7-summary.html)
