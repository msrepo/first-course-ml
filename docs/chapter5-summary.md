---
title: Chapter 5 summary, classification
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_SVG"></script>

# Chapter 5: Classification, a quick recap

Short revision notes. Exercises are not covered.

## The problem

Given $$N$$ training objects $$\mathbf{x}_n$$ (vectors of dimension $$D$$) with class labels $$t_n$$ (e.g. $$\{0,1\}$$, $$\{-1,1\}$$ or $$\{1,\dots,C\}$$), predict the class of a new object. It is the same setup as regression in Chapters 1 to 3, except the response is a class label instead of a real number. The binary response model of Chapter 4 is already a classifier (logistic regression). The chapter presents four algorithms, two probabilistic and two not.

## Probabilistic classifiers

They output $$P(t_{\text{new}}=c\mid\mathbf{x}_{\text{new}},\mathbf{X},\mathbf{t})$$ for every class. The values lie in $$[0,1]$$ and sum to 1. A probability says how confident the classifier is: 0.6 and 0.9 both lead to "class 1", but only one of them is a confident call.

### The Bayes classifier

Apply Bayes' rule to the class label:

$$
P(t_{\text{new}}=c\mid\mathbf{x}_{\text{new}},\ldots) =
\frac{p(\mathbf{x}_{\text{new}}\mid t_{\text{new}}=c,\ldots)\,P(t_{\text{new}}=c\mid\ldots)}
{\sum_{k=1}^{C} p(\mathbf{x}_{\text{new}}\mid t_{\text{new}}=k,\ldots)\,P(t_{\text{new}}=k\mid\ldots)} .
$$

- **Class-conditional distribution** $$p(\mathbf{x}\mid t=c)$$: one distribution per class, fitted using only that class's training objects. For real-valued data, a Gaussian whose mean and covariance are the sample mean and sample covariance of the class (maximum likelihood). A Bayesian treatment of these parameters is possible and helps when data is scarce.
- **Class prior** $$P(t=c)$$: either uniform, or the fraction of training objects in that class. Raising the prior of a rare class makes the classifier flag it more often.
- **Quirk.** With Gaussians of different spread, the class with the fastest-decaying density loses far from the data, so a different class can take over with high probability in empty regions. The probabilities do not become uncertain away from the data, unlike the logistic model averaged over parameters.

**Naive Bayes.** A full $$D$$-dimensional Gaussian has $$D + D(D+1)/2$$ parameters, which needs a lot of data. Naive Bayes assumes that, given the class, the dimensions are independent, so the class-conditional is a product of $$D$$ one-dimensional distributions: only $$2D$$ parameters for Gaussians. The cost is flexibility: no dependence within a class can be modelled (for Gaussians, the contours are axis-aligned). The resulting probability contours are often still reasonable.

**Text example (20 newsgroups).** Represent a document with a **bag of words**: a vector of counts over the vocabulary (the word order is ignored). Use a multinomial class-conditional with naive Bayes over words. The maximum-likelihood word probabilities are the word's share of all words in that class.

**Smoothing.** A word never seen in a class gets probability 0, which sends the whole document's likelihood for that class to 0. Fix it by putting a Dirichlet prior on the word probabilities and using the MAP estimate. This adds a pseudo-count to every word, acts as regularisation, and gives a quite good classifier despite its simplicity. A plot of predicted probabilities by true class shows where errors cluster, often between closely related topics.

### Logistic regression

Chapter 4's model, now seen as classification. The sigmoid is formally justified by modelling the **log-odds** linearly:

$$
\log\frac{P(t=1\mid\mathbf{x},\mathbf{w})}{P(t=0\mid\mathbf{x},\mathbf{w})} = \mathbf{w}^T\mathbf{x}
\;\Longrightarrow\;
P(t=1\mid\mathbf{x},\mathbf{w}) = \frac{1}{1+\exp(-\mathbf{w}^T\mathbf{x})}.
$$

This is a **generalised linear model**: a linear model passed through a transformation. Adding terms such as $$x_1^2$$, $$x_1x_2$$ to $$\mathbf{x}$$ gives curved decision boundaries, with the same over-fitting risk as high-order polynomials in Chapter 1.

### Beyond parametric models (a mention)

A **Gaussian process** puts a prior directly on function values, with a mean function and a covariance function saying how similar the outputs at two inputs should be. It is nonparametric: no fixed parametric family for the function.

## Non-probabilistic classifiers

They output a class label only.

### K-nearest neighbours (KNN)

No training phase. For a new object, find the $$K$$ closest training objects (any distance measure works) and take the majority class.

- Ties: use odd $$K$$ for two classes, or weight votes by distance.
- **Choosing $$K$$.** Small $$K$$ follows noise (isolated "islands" around mislabeled points), so it over-fits. Large $$K$$ over-smooths and lets the larger class take over, especially when classes are imbalanced. Choose $$K$$ by cross-validation using the 0/1 error.

### Support vector machines (SVM)

A binary classifier with a linear boundary, labels $$\{-1,+1\}$$, decision rule $$\operatorname{sign}(\mathbf{w}^T\mathbf{x}_{\text{new}}+b)$$. Choose $$\mathbf{w},b$$ to **maximise the margin**: the distance from the boundary to the closest points of either class.

- Rescale $$\mathbf{w},b$$ so the closest points satisfy $$\mathbf{w}^T\mathbf{x}+b=\pm1$$. The margin is then $$1/\|\mathbf{w}\|$$, so maximising it means minimising $$\tfrac12\|\mathbf{w}\|^2$$ subject to $$t_n(\mathbf{w}^T\mathbf{x}_n+b)\ge 1$$ for all $$n$$.
- Handle the constraints with **Lagrange multipliers** $$\alpha_n\ge0$$. Setting derivatives to zero gives $$\mathbf{w}=\sum_n\alpha_nt_n\mathbf{x}_n$$ and $$\sum_n\alpha_nt_n=0$$, and the problem becomes the **dual**: maximise $$\sum_n\alpha_n-\tfrac12\sum_{n,m}\alpha_n\alpha_mt_nt_m\mathbf{x}_n^T\mathbf{x}_m$$ with $$\alpha_n\ge0$$, a quadratic programme solved numerically.
- Prediction: $$\operatorname{sign}\!\big(\sum_n\alpha_nt_n\mathbf{x}_n^T\mathbf{x}_{\text{new}}+b\big)$$.
- **Support vectors** are the training points with $$\alpha_n>0$$ (those on the margin). Only they matter for the decision, so the solution is sparse. Removing the other points changes nothing.
- **Hard margin** requires every point on the correct side, so one stray point can move the boundary a lot. A **soft margin** adds slack $$\xi_n\ge0$$ and a penalty $$C\sum_n\xi_n$$. In the dual, the only change is an upper limit $$\alpha_n\le C$$, which caps the influence of any one point. Choose $$C$$ by cross-validation.
- **Kernels.** The data enter only through inner products, so replace $$\mathbf{x}_n^T\mathbf{x}_m$$ by a kernel $$k(\mathbf{x}_n,\mathbf{x}_m)=\phi(\mathbf{x}_n)^T\phi(\mathbf{x}_m)$$ and the boundary becomes linear in a transformed space and curved in the original one, without computing $$\phi$$. Common choices: linear, Gaussian $$\exp\{-\gamma\|\mathbf{x}_n-\mathbf{x}_m\|^2\}$$ and polynomial $$(1+\mathbf{x}_n^T\mathbf{x}_m)^p$$. The Gaussian kernel parameter $$\gamma$$ controls complexity: too small gives a boundary that is too simple, too large gives one that is too wiggly. $$C$$ and $$\gamma$$ interact, so tune them together. Many algorithms (for example KNN, via distances written with inner products) can be kernelised.

## Assessing performance

Take predictions $$\hat t_n$$ on test objects with true labels $$t_n^*$$.

- **Accuracy and 0/1 loss.** The fraction wrong. It is misleading with imbalanced classes: if 80% of objects are in class 1, always predicting class 1 gives a loss of 0.2.
- **Sensitivity and specificity.** Count true positives (TP), true negatives (TN), false positives (FP) and false negatives (FN). Sensitivity is TP/(TP + FN), the share of real positives found. Specificity is TN/(TN + FP), the share of real negatives found. Which one matters more depends on the application (for a rare disease, missing a case is worse than a false alarm).
- **ROC curve and AUC.** Many classifiers output a real score that is then thresholded. Sweep the threshold and plot sensitivity against 1 − specificity. A curve near the top-left corner is good. The area under it (AUC) is 1 for a perfect classifier and 0.5 for random guessing. It handles imbalance but is defined for two classes. For several classes, treat each class against the rest.
- **Confusion matrix.** A table of predicted class against true class. It generalises to many classes, and big off-diagonal entries show which classes get confused, which may suggest more data or merging classes.

## Generative and discriminative

- **Generative** classifiers model each class and assign a new object to the model that suits it best (the Bayes classifier).
- **Discriminative** classifiers model the boundary between classes directly (logistic regression, SVM).

## Take-aways

1. Probabilistic classifiers give class probabilities and confidence. Non-probabilistic ones give labels only.
2. Bayes classifier: class-conditional densities times class priors. Naive Bayes trades flexibility for far fewer parameters, and smoothing stops zero counts from vetoing a class.
3. Logistic regression models the log-odds linearly.
4. KNN and SVM both need a complexity setting ($$K$$; $$C$$ and the kernel parameter) chosen by cross-validation.
5. The SVM maximises the margin, depends only on the support vectors, and kernels give curved boundaries.
6. Judge classifiers with measures suited to the task (sensitivity, specificity, AUC, confusion matrices), not only accuracy.
