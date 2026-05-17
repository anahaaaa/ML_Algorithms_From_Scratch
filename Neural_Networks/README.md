# 🧠 Two Layer Neural Network from Scratch using NumPy

This project implements a **Two Layer Neural Network completely from scratch using Python and NumPy** without using deep learning frameworks like TensorFlow or PyTorch.

It is designed to help beginners understand how neural networks actually work internally through mathematical implementation of:

* Forward Propagation
* Backpropagation
* Weight Updates
* Activation Functions
* Gradient Descent

---

# 🚀 What is a Neural Network?

A Neural Network is a machine learning model inspired by the human brain. It learns patterns from data using interconnected layers of neurons.

This project implements:

* Input Layer
* Hidden Layer
* Output Layer

Hence the name:

## 🎯 Two Layer Neural Network

---

# 🏗️ Architecture

```text id="sxd6wg"
Input Layer → Hidden Layer → Output Layer
```

The network learns by:

1. Making predictions
2. Calculating error
3. Adjusting weights using backpropagation
4. Repeating the process until loss decreases

---

# 🧮 Mathematical Intuition

## Forward Propagation

Prediction is calculated using:

## Hidden Layer

Z^{[1]} = W^{[1]}X + b^{[1]}

Apply activation function:

A^{[1]} = \sigma(Z^{[1]})

## Output Layer

Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}

Final prediction:

A^{[2]} = \sigma(Z^{[2]})

---

# 🧠 How Backpropagation Works

Backpropagation computes gradients of the loss function and updates weights to minimize prediction error.

Steps:

1. Compute prediction
2. Calculate loss
3. Compute gradients
4. Update weights and biases
5. Repeat for multiple iterations

---

# 🧾 Code Explanation

## 1. Initialize Parameters

```python id="z2eb0d"
W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros((hidden_size, 1))

W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))
```

Weights are initialized with small random values while biases are initialized to zero.

---

# 2. Sigmoid Activation Function

```python id="h0e4q9"
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

The sigmoid function converts values into probabilities between 0 and 1.

---

# 3. Forward Propagation

```python id="1chwtm"
Z1 = np.dot(W1, X) + b1
A1 = sigmoid(Z1)

Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)
```

The network computes predictions layer by layer.

---

# 4. Compute Loss

```python id="0tmy2f"
loss = -np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2))
```

Binary Cross Entropy Loss measures prediction error.

---

# 5. Backpropagation

```python id="mjlwmr"
dZ2 = A2 - y
dW2 = np.dot(dZ2, A1.T)

dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))
dW1 = np.dot(dZ1, X.T)
```

Gradients are calculated using the chain rule.

---

# 6. Update Parameters

```python id="w2yihn"
W1 -= learning_rate * dW1
b1 -= learning_rate * db1

W2 -= learning_rate * dW2
b2 -= learning_rate * db2
```

Weights and biases are updated using Gradient Descent.

---

# 7. Train the Neural Network

```python id="l4lc4m"
for i in range(iterations):
    forward_propagation()
    backpropagation()
    update_parameters()
```

The model learns by repeating the process multiple times.

---

# 📦 How to Use This Model

```python id="d0mqj4"
nn = NeuralNetwork()

nn.fit(X_train, y_train)

predictions = nn.predict(X_test)
```

---

# 📊 Concepts Covered

* Neural Networks
* Forward Propagation
* Backpropagation
* Gradient Descent
* Weight Initialization
* Activation Functions
* Binary Classification
* Loss Functions

---

# 🧪 Test Your Understanding

Try experimenting with:

* Different learning rates
* More hidden neurons
* More layers
* Different activation functions
* Different datasets

You can also:

* Visualize loss curves
* Plot decision boundaries
* Compare performance with sklearn

---

# 📚 Recommended Topics to Learn Next

* Deep Neural Networks
* ReLU Activation
* Softmax Function
* Dropout Regularization
* Convolutional Neural Networks (CNNs)
* PyTorch
* TensorFlow

---

# ❤️ Why Build This?

Building Neural Networks from scratch helps you understand:

* How deep learning actually works internally
* Why backpropagation is important
* How models learn from data
* The mathematics behind AI systems

This gives much deeper understanding than simply using libraries.

---

# 🔧 Requirements

* Python 3.x
* NumPy
* Matplotlib
* Jupyter Notebook

Install dependencies using:

```bash id="x8jv59"
pip install numpy matplotlib notebook
```

---

# ▶️ Run the Notebook

```bash id="r78i6k"
jupyter notebook
```

Then open:

```text id="5o7vth"
2LayerNeuralNetwork.ipynb
```

---

# 📄 License

This project is open-source and available for educational use.
