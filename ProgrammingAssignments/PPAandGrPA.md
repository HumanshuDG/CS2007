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
> Write a function `hadamard(u, v)` which takes two vectors $u$ and $v$ as input and returns hadamard product of $u$ and $v$.
> Hadamard product is obtained via multiplying two matrices/vectors elementwise.
> **Inputs:** Two vectors $u$ and $v$.
> **Output:** Hadamard product of $u$ and $v$ if the dimensions of $u$ and $v$ are consistent, otherwise $None$.
```
import numpy as np
def hadamard(u, v):
	if u.shape != v.shape:
		return None
	return u * v
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
