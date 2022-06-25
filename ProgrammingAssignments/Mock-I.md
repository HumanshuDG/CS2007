# Mock OPE - I
### Mock - 1
>	You are provided with an input feature matrix $X$, label $y$ and weight $w$. <BR>
	The size of $X$, $y$ and $w$ are fixed to `(100 x 5)`, `(100 x 1)` and `(6 x 1)`. <BR>
	You have to add a dummy feature to X. <BR>
	Write a loss function named `loss` to compute ridge and lasso regression losses for the given data. <BR>
	This function should take the feature matrix, label vector, weight vector and regularization rate as inputs and return the minimum of two loss values as output.
```
import numpy as np
import random
def loss(X, y, w, lr):
    X = np.column_stack((np.ones(X.shape[0]), X))
    loss = X @ w - y
    lasso = 0.5 * loss.T @ loss + 0.5 * lr * w.T @ w
    ridge = 0.5 * loss.T @ loss + 0.5 * lr * np.sum(np.abs(w))
    return min(lasso, ridge)
```

### Mock - 2
> Researchers Cook and Weisberg (1999) measured and recorded the age (in years) and length (in mm)  of some randomly sampled 'Bluegills' fishes from Lake Mary in Minnesota. They were primarily interested in learning how the length of a bluegill fish is related to its age. Their observation by plotting the length vs age suggests that there is positive trend in the data, however, it doesn't appear to be quite linear. It appears as if there is some kind of nonlinear relationship.
> - You are provided with x (age of the fish), y (length of the fish) and test (age for which length is unknown). x and y are numpy arrays of shape 78 x 1 and test is a numpy array of size 1 X 1.
> - You have to apply second degree polynomial transformation on x.
> - Using only numpy, implement a linear regression model by stochastic gradient descent procedure on the transformed data with 300 iterations, initial weights as zero and learning rate as 0.001. <BR>
	
> What is the length of a test bluegill fish that is randomly selected based on its age? <BR>
> - Define a function named `output` which accepts $x$, $y$, test as inputs and returns new predicted length (float) as output.
```
import numpy as np
import itertools
import functools
import random
def combination(x, degree):
    return itertools.combinations_with_replacement(x, degree)
def feature(items):
    return functools.reduce(lambda x, y: x * y, items)
def polynomial_transform(x, degree):
    if x.ndim == 1:
        x = x[:, None]
    x_t = x.T
    dummy = [np.ones(x.shape[0])]
    for i in range(1, degree + 1):
        for j in combination(x_t, i):
            dummy.append(feature(j))
    return np.asarray(dummy).T
def sgd(x, epoch, lr, y):
    w = np.zeros(x.shape[1])
    for i in range(epoch):
        for j in range(x.shape[0]):
            index = random.randint(0, x.shape[0] - 1)
            x_ = x[index:index + 1]
            y_ = y[index:index + 1]
            w += -lr * x_.T @ (x_ @ w - y_)
    return w
def output(x, y, test):
    X = polynomial_transform(x, 2)
    W = sgd(X, 300, 0.001, y)
    return polynomial_transform(test, 2) @ W
```

### Mock - 3
> Define a function `ConfusionMatrix(y, y_hat)` for binary classification and return a matrix in the following format: <BR>
> `[[TN, FP], [FN, TP]]`
> - Inputs:  `y`: ($1 \times n$), `y_hat`: ($1 \times n$)
> - Output: 2D NumPy array.
```
import numpy as np
def ConfusionMatrix(y_hat, y):
    TN, FP, FN, TP = 0, 0, 0, 0
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            if y[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if y[i] == 0 and y_hat[i] == 1:
                FP += 1
            else:
                FN += 1
    cm = [[TN, FP], [FN, TP]]
    return cm
```

### Mock - 4
> A
```

```

### Mock - 5
> A
```

```

### Mock - 6
> A
```

```

### Mock - 7
> A
```

```

### Mock - 8
> A
```

```

### Mock - 9
> Define a function `is_binary(y)` to check whether the label vector y belongs to binary classification or not. <BR>
> If the labels are binary then return True (i.e., state = True (Boolean))  else return False (i.e., state = False). <BR>
> All elements in y are integer numbers (not the datatype). 
```
import numpy as np
def is_binary(y):
    return len(np.unique(y)) == 2
```
