<H1 ALIGN=CENTER> Week - 0 </H1>

### PPA - 1
> Accept a positive integer $n$ as argument and print a NumPy array of zeros to the console.
```
import numpy as np
print(np.zeros(int(input())))
```

### PPA - 2
> Write a function named `mean` that accepts a matrix (NumPy array) as argument and returns the mean of the columns, which should be a vector.
```
import numpy as np
def mean(X):
    return X.mean(axis = 0)
```

### PPA - 3
> X is a $m \times n$ matrix and $W$ is a $n \times p$ matrix. $b$ is a $p$ dimensional vector. Write a function named compute that accepts $X$, $W$ and $b$ as arguments and returns the value of the following expression:
$XW + b$
```
import numpy as np
def compute(X, W, b):
    return (X @ W) + b
```

### PPA - 4
> Write a function named `predict` that accepts two vectors $w$ and $x$ as arguments. It should return the value $1$ if $w^{T} x \ge 0$ 
and $−1$ if $w^{T}$ $x < 0$ T. This value is called the label. Don't worry about what a label is now. It will become clear in week-4.
```
import numpy as np
def predict(w, x):
	if np.sum(np.transpose(w) * x) >= 0:
		return 1
	return -1
```

<H1 ALIGN=CENTER> Week - 1 </H1>

### PPA - 1
> Write a function named `vec_addition(u, v)` which takes two vectors $u$ and $v$ as input and returns their vector addition. <BR>
> **Inputs:** $u$ and $v$ are two numerical numpy arrays. <BR>
> **Output:** a vector representing sum of $u$ and $v$, if they have consistent shapes, $None$ otherwise.
```
import numpy as np
file = open("test.py", 'w')
def vec_addition(u, v):
	if len(u) != len(v):
		return None
	return u + v
```

### PPA - 2
> Write a function named `scalar_multiply(u, k)` which takes a vector $u$ and a scalar $k$ as input and returns $k$ times $u$ as output. <BR>
> **Inputs:** A vector $u$ and a scalar $k$. <BR>
> **Output:** Returns a vector which is $k$ times $u$.
```
import numpy as np
def scalar_multiply(u, k):
	return u * k
```

### PPA - 3
> Write a function `hadamard(u, v)` which takes two vectors $u$ and $v$ as input and returns hadamard product of $u$ and $v$. <BR>
> Hadamard product is obtained via multiplying two matrices/vectors elementwise. <BR>
> **Inputs:** Two vectors $u$ and $v$. <BR>
> **Output:** Hadamard product of $u$ and $v$ if the dimensions of $u$ and $v$ are consistent, otherwise $None$.
```
import numpy as np
def hadamard(u, v):
	if u.shape != v.shape:
		return None
	return u * v
```

### GrPA - 1
> Write a function `dot_product(u, v)` which take two vectors $u$ and $v$ as input and returns inner product of $u$ and $v$ as output. <BR>
> **Inputs:** vectors $u$ and $v$. <BR>
> **Output:** dot product of $u$ and $v$ if $u$ and $v$ have consistent dimension otherwise $None$.
```
import numpy as np
def dot_product(u, v):
	if u.shape == v.shape:
		return np.dot(u, v)
	return None
```

### GrPA - 2
> Write a function `add_constant(u, k)` which adds a constant $k$ to each element of $u$. <BR>
> **Inputs:** $u$ is a vector, $k$ is a scalar. <BR>
> **Output:** $C$ is a vector
```
import numpy as np
def add_constant(u, v):
	return u + v
```

### GrPA - 3
> Write a function `add_matrix(X, Y)` to add two matrices $X$ and $Y$. <BR>
> **Inputs:** $X$ is a matrix, $Y$ is a matrix. <BR>
> **Output:** $Z$ is a matrix representing ($X + Y$) if $X$ and $Y$ have consistent dimension otherwise $None$.
```
import numpy as np
def add_matrix(X, Y):
    try:
        return X + Y
    except:
        return None
```

<H1 ALIGN=CENTER> Week - 2 </H1>

### PPA - 1
> Write a function add_one(X) to include first column with all elements 1 in X. <BR>
> **Inputs:** A matrix $X$. <BR>
> **Output:** updated matrix with a column having elements 1 added as first column to $X$.
```
import numpy as np
def add_one(X):
	return np.column_stack((np.ones(X.shape[0]), X))
```

### PPA - 2
> Write a function `multiply(X, w)` to Multiply feature matrix($X$) and weight vector($w$) after addition of dummy feature to feature matrix. <BR>
> **Inputs:** Feature matrix $X$ and weight vector $w$. <BR>
> **Output:** Product of $X$ and $w$ after adding dummy feature to feature matrix $X$. If the dimensions are not consistent return $None$.
```
import numpy as np
def multiply(X, w):
	a = np.column_stack((np.ones(X.shape[0]), X))
	if a.shape[-1] != w.shape[0]:
		return None
	return a @ w
```

### PPA - 3
> Write a function `loss(X, w, y)` which takes feature matrix($X$), weight vector($w$) and output label vector($y$) and returns sum squared loss while implementing regression model. <BR>
> ***Note: Do necessary preprocessing of $X$*** <BR>
> **Inputs:** Feature matrix $X$ and weight vector $w$ and output label vector $y$. <BR>
> **Output:** sum squared loss if dimensions of inputs are consistent, otherwise $None$.
```
import numpy as np
def loss(X, w, y):
	a = np.column_stack((np.ones(X.shape[0]), X))
	if a.shape[-1] != w.shape[0]:
		return None
	f = a @ w
	e = f - y
	l = 0.5 * (np.transpose(e) @ e)
	return l
```

### GrPA - 1
> Write a function `compatibility(X, w)` to find whether $X$ and $w$ can be multiplied or not. <BR>
> **Inputs:** Feature matrix $X$ and weight vector $w$ <BR>
> **Output:** return `C` if $X$ and $w$ can be multiplied otherwise $None$. <BR>
> **Note: Preprocess $X$, if necessary.**
```
import numpy as np
def compatibility(X, w):
	if (X.shape[-1] + 1) == w.shape[0]:
		return "C"
	return None
```

### GrPA - 2
> Write a function `gradient(X, w, y)` to calculate gradient of loss function w.r.t weight vector given that $X$ is the feature matrix, $w$ is the weight vector and $y$ is the output vector. <BR>
> **Inputs:** Feature matrix $X$, weight vector $w$, and output label vector $y$. <BR>
> **Output:** gradient if dimesnsions of inputs are consistent, otherwise $None$. <BR>
> **Note: Preprocess $X$, if necessary.**
```
import numpy as np
def gradient(X, w, y):
	X = np.column_stack((np.ones(X.shape[0]), X))
	if X.shape[-1] == w.shape[0]:
		b = X @ w - y
		return(X.T @ b)
	return None
```

### GrPA - 3
> Write a function `weight_update(X, w, y, lr)` to get updated weight after one iteration of gradient descent given that $X$ is the feature matrix, $w$ is the weight vector and $y$ is the output label vector. <BR>
> **Inputs:** Feature matrix $X$, weight vector $w$, output label vector $y$ and learning rate $lr$. <BR>
> **Output:** weight updates after gradient calculation if dimensions of inputs are consistent, otherwise $None$. <BR>
> **Note: Do necessary preprocessing of $X$.**
```
import numpy as np
def weight_update(X, w, y, lr):
	X = np.column_stack((np.ones(X.shape[0]), X))
	if X.shape[-1] == w.shape[0]:
		b = X @ w - y
		grad = X.T @ b
		return w - lr * grad
	return None
```

<H1 ALIGN=CENTER> Week - 3 </H1>

### PPA - 1
> Consider input feature vector $X$. The shape of $X$ is fixed ($1 x\times 3$) and the degree of the polynomial is a random number (between 2 to 10). <BR>
> Write a function named `polynomial_transform` which takes the feature vector and degree of the polynomial as inputs and returns the transformed polynomial feature as output as a numpy array.
```
import numpy as np
import random
import itertools
import functools
def polynomial_transform(x, degree):
	f = np.ones(len(x))
	for i in range(1, degree - 1):
		f.append()
```

### PPA - 2
> Consider input feature matrix $X$, label $y$ and weight $w$. The size of $X$, $y$ and $w$ are fixed to ($10 \times 3$), ($10 \times 1$) and ($4 \times 1$). <BR>
> Write a loss function named `ridge_loss` to compute ridge regression loss. This function should take the feature matrix ($X$), label vector ($y$), weight vector($w$) and regularization rate ($l$) as inputs and return the loss value as output. <BR>
**Note: Add a dummy feature to $X$.**
```

```

### GrPA - 1
> Government of India is taking several steps to ensure that we are well prepared to face the challenges and threats posed by COVID-19. With active support of citizens of India, Goverment have been able to mitigate the spread of the virus so far. One of the most important factors in the fight with the virus is to vaccinate maximum people and enable them to take precautions as per the advisories being issued by different Ministries. <BR>
> Total 50 month's data and total vaccination in each month is given. <BR>
> After initial data exploration you got to know that the given data don't follow linear model. <BR>
> Now You have to define function `model_error` which will fit the given data on linear regression model with appropriate degree and return RMSE error.
```
import numpy as np
import random
from sklearn.linear_model import LinearRegression
def model_error(X, y,degree):
	transformed_features = polynomial_transform(X, degree)
	polyreg = LinearRegression()
	w_nonlin = polyreg.fit(transformed_features, y)
	y_pred = w_nonlin.predict(polynomial_transform(X, degree))
	training_error = y_pred - y
	train_loss = (1 / 2) * (np.transpose(training_error) @ training_error) 
	rmse = np.sqrt((2 / X.shape[0]) * train_loss)
	return rmse
```

### GrPA - 2
> Consider input feature matrix $X$ having size of ($20 \times 3$).
> **You do not have to generate these values.**
> Write a function named `additional_vector` which will creates a list of size $20$ having $1$ at even place and $0$ at odd place (e.g - `[0, 1, 0, 1]). Then later it returns new updated matrix by stacking $X$ and new vector columnwise.
```
import numpy as np
import random
def additional_vector(X):
	dummy=[]
	for i in range(len(X)):
		if i % 2 == 0:
			dummy.append(0)
	else:
		dummy.append(1)
	updated_X = np.column_stack((X, dummy))
	return updated_X
```

### GrPA - 3
> Consider input feature matrix $X$, label $y$ and weight $w$. The size of $X$, $y$ and $w$ are fixed to ($20 \times 3$), ($20 \times 1$) and ($3 \times 1$) respectively and these will have random values. <BR>
> **You do not have to generate these values.** <BR>
> Write a function named `mean_abs` to compute mean absolute error. This function should take the feature matrix, label vector, weight vector as inputs and return the mean absolute value as output.
```
import numpy as np
import random
def mean_abs(X, y, w):
	predicted_y = X @ w
	error = y - predicted_y
	abs_error = abs(error)
	mean_matrix = (1 / len(X)) * (abs_error)
	mean_sum = np.sum(np.sum(mean_matrix, axis = 1))
	return mean_sum
```


<H1 ALIGN=CENTER> Week - 4 </H1>

### PPA - 1
> A
```

```

### PPA - 2
> A
```

```

### PPA - 3
> A
```

```

### GrPA - 1
> A
```

```

### GrPA - 2
> A
```

```

### GrPA - 3
> A
```

```
