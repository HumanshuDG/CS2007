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


<H1 ALIGN=CENTER> Week - 2 </H1>

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


<H1 ALIGN=CENTER> Week - 3 </H1>

### PPA - 1
> A
```

```

### PPA - 2
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
