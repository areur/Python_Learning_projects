# Project: Analytical Gradient Calculator. 

Derive the gradient of a single neuron's loss function and verify it in code

## Project Structure
```
AnalyticalGradientCalculator/
│
├── src/
│   ├── activations.py          # Forward activations & their explicit derivatives (e.g., Sigmoid, ReLU)
│   ├── losses.py               # Loss functions & their forward/backward mathematical evaluations
│   └── neuron.py               # neuron class w/ parameters, forward pass, & analytic gradients

├── utils/
│   └── gradient_checker.py     # Finite-difference numerical approximation script to verify analytic math
│
├── main.py                     # Primary execution script evaluating gradients at specific coordinates
└── requirements.txt            # Project dependencies (NumPy)
```
## Derivation
Linear Unit: $z = w^{\top} x + b$

Activation Function: $\hat{y} = \sigma(z)$

Loss function: $L(e) = \frac{1}{2}(e)^2$ where $e=\hat{y}-y$

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial e}\frac{\partial e}{\partial \hat{y}}\frac{\partial \hat{y}}{\partial z}\frac{\partial z}{\partial w}
$$

$$
\frac{\partial L}{\partial e} = e \\\\ \frac{\partial e}{\partial \hat{y}} = 1 \\\\ \frac{\partial \hat{y}}{\partial z} = \sigma'(z) \\\\ \frac{\partial z}{\partial w} = \frac{\partial}{\partial w}([w_{1} x_{1}+ w_{2} x_{2} +...+w_{D} x_{D}]) = x^\top
$$

$$
\frac{\partial L}{\partial w} = (\hat{y}-y) \cdot 1 \cdot \sigma'(z) \cdot x^\top
$$
