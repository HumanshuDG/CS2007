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
    lasso = 0.5 * loss.T @ loss + 0.5 * lr * w.T @w
    ridge = 0.5 * loss.T @ loss + 0.5 * lr * np.sum(np.abs(w))
    return min(lasso, ridge)
```

### Mock - 2
> A
```

```

### Mock - 3
> A
```

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
> A
```

```
