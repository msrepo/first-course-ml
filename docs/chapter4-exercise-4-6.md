---
title: Exercise 4.6, Poisson regression and the MAP solution
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_SVG"></script>

# Exercise 4.6: Poisson regression and the MAP solution

Chapter 4 notes.

**The exercise.** We observe $$N$$ attribute vectors $$\mathbf{x}_1,\dots,\mathbf{x}_N$$ with associated integer counts $$t_1,\dots,t_N$$. A Poisson likelihood suits count data:

$$
p(t_n\mid\mathbf{x}_n,\mathbf{w})=\frac{V_n^{\,t_n}\exp\{-V_n\}}{t_n!},\qquad V_n=\exp(\mathbf{w}^T\mathbf{x}_n).
$$

With a Gaussian prior on $$\mathbf{w}$$, derive the gradient and the Hessian needed to run Newton–Raphson for the MAP weights.

## The model in words

- **$$V_n$$ is the rate.** For a Poisson variable, the mean and the variance both equal $$V_n$$. So $$V_n$$ is the model's predicted count for observation $$n$$.
- **Why the exponential.** A rate must be positive, and $$\mathbf{w}^T\mathbf{x}_n$$ can be any real number. Passing it through $$\exp$$ makes $$V_n$$ positive. The log of the rate is linear in the attributes, so each unit of $$\mathbf{w}^T\mathbf{x}_n$$ multiplies the expected count by $$e$$.
- **The parallel with Chapter 4's classifier.** In the binary model, a sigmoid turned the linear score into a probability between 0 and 1. Here an exponential turns it into a positive rate. Both are generalised linear models: a linear score pushed through a function chosen to suit the target.

The prior is the one used in the chapter, $$p(\mathbf{w}\mid\sigma^2)=\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})$$.

## Step 1: the log of g

As in the chapter, $$g(\mathbf{w};\mathbf{X},\mathbf{t},\sigma^2)=p(\mathbf{t}\mid\mathbf{X},\mathbf{w})\,p(\mathbf{w}\mid\sigma^2)$$ is proportional to the posterior. For independent observations the log likelihood is a sum over $$n$$:

$$
\log g(\mathbf{w})=\sum_{n=1}^{N}\big[t_n\log V_n-V_n-\log t_n!\big]-\frac{\mathbf{w}^T\mathbf{w}}{2\sigma^2}+\text{const}.
$$

Two simplifications make the rest easy:
- Since $$V_n=\exp(\mathbf{w}^T\mathbf{x}_n)$$, we have $$\log V_n=\mathbf{w}^T\mathbf{x}_n$$. The log and the exponential cancel, so $$t_n\log V_n=t_n\,\mathbf{w}^T\mathbf{x}_n$$.
- The term $$\log t_n!$$ does not contain $$\mathbf{w}$$, so it drops out of both derivatives.

The prior term $$-\mathbf{w}^T\mathbf{w}/(2\sigma^2)$$ comes from $$\boldsymbol{\Sigma}^{-1}=\mathbf{I}/\sigma^2$$ in the Gaussian log density. $$\mathbf{w}^T\mathbf{w}=\sum_jw_j^2$$ is the squared length of the weight vector, so the prior penalises large weights. This leaves

$$
\log g(\mathbf{w})=\sum_{n=1}^{N}\big[t_n\,\mathbf{w}^T\mathbf{x}_n-\exp(\mathbf{w}^T\mathbf{x}_n)\big]-\frac{\mathbf{w}^T\mathbf{w}}{2\sigma^2}+\text{const}.
$$

## Step 2: the gradient

Differentiate term by term with respect to $$\mathbf{w}$$:
- $$\dfrac{\partial}{\partial\mathbf{w}}\,t_n\,\mathbf{w}^T\mathbf{x}_n=t_n\,\mathbf{x}_n$$.
- $$\dfrac{\partial}{\partial\mathbf{w}}\exp(\mathbf{w}^T\mathbf{x}_n)=\exp(\mathbf{w}^T\mathbf{x}_n)\,\mathbf{x}_n=V_n\,\mathbf{x}_n$$. This is the chain rule (Comment 4.2): the exponential stays, and we multiply by the derivative of its argument, which is $$\mathbf{x}_n$$.
- $$\dfrac{\partial}{\partial\mathbf{w}}\Big(-\dfrac{\mathbf{w}^T\mathbf{w}}{2\sigma^2}\Big)=-\dfrac{\mathbf{w}}{\sigma^2}$$, by the rule $$\partial(\tfrac12\mathbf{w}^T\mathbf{A}\mathbf{w})/\partial\mathbf{w}=\mathbf{A}\mathbf{w}$$ with $$\mathbf{A}=\mathbf{I}/\sigma^2$$.

Adding them up:

$$
\boxed{\;\frac{\partial\log g}{\partial\mathbf{w}}=\sum_{n=1}^{N}\mathbf{x}_n\,(t_n-V_n)-\frac{1}{\sigma^2}\mathbf{w}\;}
$$

**Reading it.** Each observation pushes $$\mathbf{w}$$ along its own attribute vector $$\mathbf{x}_n$$, by an amount equal to the miss $$t_n-V_n$$ between the observed and the predicted count. If the model under-predicts ($$t_n>V_n$$), the push increases the rate for inputs like $$\mathbf{x}_n$$. The prior pulls the other way, toward zero. This has exactly the structure of the binary model's gradient, $$\sum_n\mathbf{x}_n(t_n-P_n)-\mathbf{w}/\sigma^2$$, with the predicted probability replaced by the predicted rate.

## Step 3: the Hessian

Differentiate the gradient with respect to $$\mathbf{w}^T$$. Only $$V_n$$ depends on $$\mathbf{w}$$ inside the sum, and

$$
\frac{\partial V_n}{\partial\mathbf{w}^T}=V_n\,\mathbf{x}_n^T .
$$

The term $$\mathbf{x}_n t_n$$ is constant. Using $$-\mathbf{x}_n\,\partial V_n/\partial\mathbf{w}^T=-V_n\,\mathbf{x}_n\mathbf{x}_n^T$$, and $$\partial(-\mathbf{w}/\sigma^2)/\partial\mathbf{w}^T=-\mathbf{I}/\sigma^2$$:

$$
\boxed{\;\frac{\partial^2\log g}{\partial\mathbf{w}\,\partial\mathbf{w}^T}=-\sum_{n=1}^{N}V_n\,\mathbf{x}_n\mathbf{x}_n^T-\frac{1}{\sigma^2}\mathbf{I}\;}
$$

**Unlike Exercise 4.1,** this Hessian contains $$\mathbf{w}$$, through $$V_n=\exp(\mathbf{w}^T\mathbf{x}_n)$$. It is the analogue of the binary model's Hessian, with $$P_n(1-P_n)$$ replaced by $$V_n$$.

## Step 4: why Newton–Raphson works here

For any non-zero vector $$\mathbf{d}$$,

$$
\mathbf{d}^T\mathbf{H}\,\mathbf{d}=-\sum_nV_n\,(\mathbf{x}_n^T\mathbf{d})^2-\frac{\mathbf{d}^T\mathbf{d}}{\sigma^2}<0,
$$

because every $$V_n>0$$, every square is non-negative, and the last term is strictly negative. So $$\mathbf{H}$$ is negative definite for every $$\mathbf{w}$$. Therefore $$\log g$$ is concave, it has at most one stationary point, and wherever Newton–Raphson converges, that point is the global MAP. The prior's term is what guarantees this even when the data alone do not (for example with more weights than observations).

## Step 5: the Newton–Raphson update

Equation 4.6 is $$\mathbf{w}'=\mathbf{w}-\mathbf{H}^{-1}\nabla\log g$$. Substituting:

$$
\mathbf{w}'=\mathbf{w}+\Big(\sum_nV_n\mathbf{x}_n\mathbf{x}_n^T+\frac{1}{\sigma^2}\mathbf{I}\Big)^{-1}\Big(\sum_n\mathbf{x}_n(t_n-V_n)-\frac{1}{\sigma^2}\mathbf{w}\Big).
$$

**In matrix form.** Let $$\mathbf{X}$$ be the $$N\times D$$ matrix with rows $$\mathbf{x}_n^T$$, $$\mathbf{v}=(V_1,\dots,V_N)^T$$ and $$\boldsymbol{\Lambda}=\mathrm{diag}(V_1,\dots,V_N)$$. Then the gradient is $$\mathbf{X}^T(\mathbf{t}-\mathbf{v})-\mathbf{w}/\sigma^2$$ and the Hessian is $$-(\mathbf{X}^T\boldsymbol{\Lambda}\mathbf{X}+\mathbf{I}/\sigma^2)$$.

**The algorithm.**
1. Start with $$\mathbf{w}=\mathbf{0}$$, where every $$V_n=1$$.
2. Compute $$V_n=\exp(\mathbf{w}^T\mathbf{x}_n)$$ for all $$n$$, then the gradient and the Hessian.
3. Update $$\mathbf{w}$$ with the formula above.
4. Stop when $$\mathbf{w}$$ stops changing. The result is $$\hat{\mathbf{w}}$$.

A caution: the exponential can overflow if $$\mathbf{w}^T\mathbf{x}_n$$ gets large, which can happen in the first steps from a poor start. Starting from zero and taking Newton steps usually behaves, but rescaling the attributes helps.

## A one-weight worked example

Take one weight, $$x_n=1$$ for both of two points, counts $$t=(2,4)$$ and $$\sigma^2=10$$. Then $$V_n=e^{w}$$ for both points.

- **Gradient:** $$\sum_n(t_n-e^w)-w/10$$.
- **Hessian:** $$-2e^w-1/10$$.
- **From $$w=0$$:** the gradient is $$(2-1)+(4-1)=4$$ and the Hessian is $$-2.1$$, so $$w'=0-4/(-2.1)\approx1.905$$.

Repeating the steps, the iterates settle at $$\hat w\approx1.080$$ after about five steps.

**Checks.**
- **No prior** ($$\sigma^2\to\infty$$). Setting the gradient to zero gives $$e^{\hat w}=\tfrac1N\sum_nt_n$$, the mean count, so $$\hat w=\log(3)\approx1.099$$ here. The prior pulls the MAP estimate slightly toward zero, which is why $$1.080<1.099$$.
- **Shapes.** The gradient is $$D\times1$$ and the Hessian is $$D\times D$$.
- **Signs.** The gradient points toward larger $$\mathbf{w}$$ where counts exceed predictions, and the Hessian is negative everywhere.

## What comes next

With $$\hat{\mathbf{w}}$$ in hand, the Laplace approximation (the next exercise) is $$\mathcal{N}(\hat{\mathbf{w}},\boldsymbol{\Sigma})$$ with $$\boldsymbol{\Sigma}=-\mathbf{H}^{-1}$$ evaluated at $$\hat{\mathbf{w}}$$, that is $$\boldsymbol{\Sigma}=\big(\mathbf{X}^T\hat{\boldsymbol{\Lambda}}\mathbf{X}+\mathbf{I}/\sigma^2\big)^{-1}$$. Metropolis–Hastings needs only $$g$$ itself.
