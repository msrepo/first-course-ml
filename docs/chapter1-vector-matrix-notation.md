---
title: Vector and matrix notation, and differentiating with it
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML"></script>

# Vector and matrix notation, and differentiating with it

Chapter 1 notes.

## 1. Notation

A **vector** is a column of numbers. Bold lowercase letters name vectors, and a superscript $$T$$ turns a column into a row (transpose).

- Parameters: $$\mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}$$
- Targets (the values to predict): $$\mathbf{t} = \begin{bmatrix} t_1 \\ \vdots \\ t_N \end{bmatrix}$$

A **matrix** is a grid of numbers. Bold uppercase letters name matrices. For a straight-line model, each row of $$\mathbf{X}$$ is one data point, written as a 1 followed by its input value $$x_n$$:

$$
\mathbf{X} = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_N \end{bmatrix}
$$

The column of ones is there so that $$w_0$$ has something to multiply.

**Matrix times vector.** The product $$\mathbf{X}\mathbf{w}$$ is a vector with one entry per data point. Entry $$n$$ is $$w_0 + w_1 x_n$$, which is the model's prediction for that point. Seen column by column, the whole output is a weighted sum of the two columns of $$\mathbf{X}$$:

$$
\mathbf{X}\mathbf{w} = w_0 \begin{bmatrix} 1 \\ \vdots \\ 1 \end{bmatrix} + w_1 \begin{bmatrix} x_1 \\ \vdots \\ x_N \end{bmatrix}
$$

So $$w_0$$ adds a constant shift, and $$w_1$$ adds a multiple of the input column.

**Inner product.** For two vectors of the same length, $$\mathbf{a}^T\mathbf{b} = \sum_n a_n b_n$$ is a single number. In particular $$\mathbf{a}^T\mathbf{a}$$ is the sum of the squares of the entries.

## 2. The loss in this notation

The residual vector is $$\mathbf{t} - \mathbf{X}\mathbf{w}$$: one error per data point. The sum of squared errors is the residual's inner product with itself, so the average squared-error loss is

$$
L = \frac{1}{N}\,(\mathbf{t} - \mathbf{X}\mathbf{w})^T(\mathbf{t} - \mathbf{X}\mathbf{w}).
$$

Multiplying out (transpose of a product reverses the order, $$(\mathbf{X}\mathbf{w})^T = \mathbf{w}^T\mathbf{X}^T$$):

$$
L = \frac{1}{N}\left( \mathbf{t}^T\mathbf{t} \;-\; 2\,\mathbf{w}^T\mathbf{X}^T\mathbf{t} \;+\; \mathbf{w}^T\mathbf{X}^T\mathbf{X}\,\mathbf{w} \right).
$$

The middle term has $$\mathbf{w}$$ appearing once, and the last has it appearing twice. This is the vector version of $$c - 2bw + aw^2$$.

## 3. Differentiating with respect to a vector

For a function $$f(\mathbf{w})$$ that returns one number, the **gradient** $$\dfrac{\partial f}{\partial \mathbf{w}}$$ is the vector of its partial derivatives, one per parameter:

$$
\frac{\partial f}{\partial \mathbf{w}} = \begin{bmatrix} \partial f/\partial w_0 \\ \partial f/\partial w_1 \end{bmatrix}.
$$

It points in the direction in which $$f$$ increases fastest. At a minimum it is the zero vector.

Two rules cover everything needed here. Each has a one-variable twin:

| Vector rule | One-variable twin |
|---|---|
| $$\dfrac{\partial}{\partial \mathbf{w}}\,\mathbf{b}^T\mathbf{w} = \mathbf{b}$$ | $$\dfrac{d}{dw}\,bw = b$$ |
| $$\dfrac{\partial}{\partial \mathbf{w}}\,\mathbf{w}^T\mathbf{A}\mathbf{w} = 2\mathbf{A}\mathbf{w}$$ (for symmetric $$\mathbf{A}$$) | $$\dfrac{d}{dw}\,aw^2 = 2aw$$ |

Why the first rule holds: $$\mathbf{b}^T\mathbf{w} = b_0 w_0 + b_1 w_1$$, so its partial derivatives are $$b_0$$ and $$b_1$$, which stack into $$\mathbf{b}$$. With $$\mathbf{A}=\mathbf{I}$$ the second rule gives $$\partial(\mathbf{w}^T\mathbf{w})/\partial\mathbf{w} = 2\mathbf{w}$$, which you can check the same way from $$w_0^2 + w_1^2$$.

## 4. Putting it together: the least-squares solution

Differentiate the expanded loss. The first term has no $$\mathbf{w}$$ and drops out. The second uses rule 1 with $$\mathbf{b}=\mathbf{X}^T\mathbf{t}$$. The third uses rule 2 with $$\mathbf{A}=\mathbf{X}^T\mathbf{X}$$, which is symmetric.

$$
\frac{\partial L}{\partial \mathbf{w}} = \frac{2}{N}\left( \mathbf{X}^T\mathbf{X}\,\mathbf{w} - \mathbf{X}^T\mathbf{t} \right).
$$

Setting this to zero gives the **normal equations** and their solution:

$$
\mathbf{X}^T\mathbf{X}\,\mathbf{w} = \mathbf{X}^T\mathbf{t}
\quad\Longrightarrow\quad
\widehat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{t}.
$$

This one line replaces the separate formulas for $$w_1$$ and $$w_0$$, and it keeps working unchanged when $$\mathbf{X}$$ has more columns (more features).
