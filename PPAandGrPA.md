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
> **Output:** a vector representing sum of $u$ and $v$, if they have consistent shapes, `None` otherwise.
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
> **Output:** Hadamard product of $u$ and $v$ if the dimensions of $u$ and $v$ are consistent, otherwise `None`.
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
> **Output:** dot product of $u$ and $v$ if $u$ and $v$ have consistent dimension otherwise `None`.
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
> **Output:** $Z$ is a matrix representing ($X + Y$) if $X$ and $Y$ have consistent dimension otherwise `None`.
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
> **Output:** Product of $X$ and $w$ after adding dummy feature to feature matrix $X$. If the dimensions are not consistent return `None`.
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
> **Output:** sum squared loss if dimensions of inputs are consistent, otherwise `None`.
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
> **Output:** return `C` if $X$ and $w$ can be multiplied otherwise `None`. <BR>
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
> **Output:** gradient if dimesnsions of inputs are consistent, otherwise `None`. <BR>
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
> **Output:** weight updates after gradient calculation if dimensions of inputs are consistent, otherwise `None`. <BR>
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
import numpy as np
import random
def ridge_loss(X, y, w, l):
	X = np.column_stack((np.ones(X.shape[0]), X))
	return (0.5 * ((y - (X @ w)).T @ (y - (X @ w)))) + ((1 / 2) * (w.T @ w))
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
> Consider input feature matrix $X$ having size of ($20 \times 3$). <BR>
> **You do not have to generate these values.** <BR>
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
> Define a function `ConfusionMatrix(y, y_hat)` for binary classification and return a matrix in the following format: <BR>
> `[[TN,FP],[FN,TP]]` <BR>
> **Inputs:** `y`: ($1 \times n$), `y_hat`: ($1 \times n$). <BR>
> **Output:** 2D numpy array.
```
# ERROR IN PROGRAM SO JUST SUBMIT	
import numpy as np
def ConfusionMatrix(y, y_hat):
    TP = np.where((y_hat == 1 and y == 1), 1, 0).sum()
    TN = np.where((y_hat == 0 and y == 0), 1, 0).sum()
    FN = np.where((y_hat == 0 and y == 1), 1, 0).sum()
    FP = np.where((y_hat == 1 and y == 0), 1, 0).sum()
    cm = [[TN, FP], [FN, TP]]
    return cm
```

### PPA - 2
> Define a function `is_binary(y)` to check whether the label vector $y$ belongs to binary classification or not. If the labels are binary then return `True` (i.e., state = True (Boolean)) else return `False` (i.e., state = False). <BR>
> All elements in $y$ are integer numbers (not the datatype).
```
import numpy as np
def is_binary(y):
	return np.where((len(np.unique(y)) == 2), True, False)
```

### PPA - 3
> Write a function `percep_loss(X,w,y,i)` to compute the perceptron loss for all the sample in $X$ and return the loss for individual samples in a vector.
```
import numpy as np
def percep_loss(X, w, y):
    y_hat = np.where((X @ w) >= 0, 1, -1)
    samplewise_loss = (np.maximum(-1 * y_hat * y, np.zeros(y.shape[0])))
    return samplewise_loss
```

### GrPA - 1
> Write a function `OneHotEncode(y)` to convert integer labels to one hot-encoded labels.
```
import numpy as np
def OneHotEncode(y):
	encoded = np.zeros((y.size, y.max() + 1))
	encoded[np.arange(y.size), y] = 1.0
	return encoded
```

### GrPA - 2
> Implement the perceptron weight update rule. Name the function as `update(x, w, y, epoch)` to update the weight vector over $n$ epochs and returns the history of the weight updates as a matrix (or a vector if there is only one feature). <BR>
> The row represents the weight values at $i^{th}$ epoch. The zeroth row represents the weight value at epoch zero (that is the one directly passed an argument to the function). <BR>
> **Note: Keep learning rate alpha as 1.0**
```
import numpy as np
def update(x,w,y,epochs):
	history = np.zeros((epochs+1,w.size))
	for epoch in np.arange(epochs):
		for xi, target in zip(x, y):
			w += 1 * (target - np.where(xi @ w >= 0, 1, -1)) * xi
		history[epoch + 1] = w
	return history
```

### GrPA - 3
> Implement a function with a name `is_linearly_separable()` that takes in data matrix with a dummy feature ($X$) and label vector ($y$). The function returns `True` if the datapoints are linearly separable, `False` otherwise. <BR>
> **Note: The maximum number of epochs should not exceed 10.**
```
import numpy as np
def is_linearly_separable(x,y):
	def loss(features, labels, weights):
		e = np.where(features @ weights >= 0, 1, -1) - labels
		return (e.transpose() @ e) 
	w = np.zeros(x.shape[1])
	state = False
	for epoch in np.arange(epochs):
		for xi, target in zip(x, y):
			w += 0.5 * (target - np.where(xi @ w >= 0, 1, -1)) * xi
		if loss(x, y, w) == 0.0:
			state = True
	return state
```


<H1 ALIGN=CENTER> Week - 5 </H1>

### PPA - 1
> Define a function `cross_entropy(y, sigmoid_vector, w, reg_type, reg_rate)` having the following characteristics:
> 
> Input:
> - y: Actual output label vector
> - sigmoid_vector: logistic value of predicted output 
> - w: weight vector
> - reg_type: type of regularization as string, either 'l1' or 'l2'. Default 'l2'.
> - reg_rate: regularization rate. Default value 0.
> 
> Output:
> - Binary cross entropy loss(float value)
```
import numpy as np
def cross_entropy(y, sigmoid_vector, w, reg_type = "l2", reg_rate = 0):
    if reg_type == 'l1':
        loss = (-1 * np.sum(y * np.log(sigmoid_vector) + (1 - y) * np.log(1 - sigmoid_vector))) + reg_rate * np.sum(np.abs(w))
    else:
        loss = (-1 * np.sum(y * np.log(sigmoid_vector) + (1 - y) * np.log(1 - sigmoid_vector))) + reg_rate * (np.transpose(w) @ w)
    return loss
```

### PPA - 2
> Write a function `sigmoid(X)` which returns logistic function of `X`, where `X` is a numpy array.
> 
> Input:
> - A numpy array `X`.
> 
> Output:
> - Logistic function of `X`.
```
import numpy as np
def sigmoid(X):
    return 1 / (1 + np.exp(-X))
```

### GrPA - 1
> Assume that we have trained a logistic regression classifier on a dataset and have learned the weight `w`. Define a function `predict_label(X, w)` which accepts a feature matrix `X` of test samples and the weight vector `w` as arguments, and assigns labels to each of the samples based on the following conditions:
> - If the model's output is greater than or equal to 0.75, assign the predicted label as ` 1 `.
> - If the model's output is less than or equal to 0.25, assign the predicted label as ` -1 `.
> - Otherwise, assign the label as ` 0 `
> 
> The function should return the vector of predicted labels.
> 
> Use the sigmoid activation function while calculating the model's output for all the sample values in the test-set.
```
import numpy as np
def predict_label(X, w):
    z = X @ w
    sig = 1 / (1 + np.exp(-z))
    labels = (sig >= 0.25).astype(int)
    return labels
```

### GrPA - 2
> Define a function `gradient(X, y, w, reg_rate)` which can be used for optimization of logistic regression model with L2 regularization having the following characteristics:
> 
> Input:
> - `X`: Feature matrix for training data.
> - `y`: Label vector for training data.
> - `reg_rate`: regularization rate
> - `w`: weight_vector
> 
> Output:
> - A vector of gradients.
```
import numpy as np
def gradient(X, y, w, reg_rate):
    z = X @ w
    sig = 1 / (1 + np.exp(-z))
    return np.transpose(X) @ (sig-y) + reg_rate * w
```

### GrPA - 3
> Define a function `update_w(X, y, w, reg_rate, lr)` which can be used for optimization of logistic regression model with L2 regularization having following characteristics:
> 
> Input:
> - `X`: Feature matrix for training data.
> - `y`: Label vector for training data.
> - `reg_rate`: regularization rate
> - `w`: weight_vector
> - `lr`: learning rate
> 
> Output:
> - A vector of updated weights.
> 
> You need to perform exactly one update over the entire data.
```
import numpy as np
def update_w(X, y, w, reg_rate, lr):
    z = X @ w
    sig = 1 / (1 + np.exp(-z))
    G = np.transpose(X) @ (sig - y) + reg_rate * w
    return w - lr * G
```


<H1 ALIGN=CENTER> Week - 6 </H1>

### PPA - 1

> Consider a training dataset $D = { \{ x^{(i)}, y^{(i)} \} }_{i = 1}^{100}$ for a binary classification problem, where the feature vector $x = (x_1, x_2)$ is a two-dimensional binary vector, i.e., each feature is binary. The class label $y$ is indexed using $1$ and $2$. A sample feature matrix and label vector is given below:
> 
> $X = \begin{bmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}, \hspace{5pt} y = \begin{bmatrix} 1\\ 2\\ 2\\ 2 \end{bmatrix}$
> 
> Assume that the features are conditionally independent given the class labels. Train a Bernoulli Naive-Bayes classifier for this data. Specifically, estimate the following parameter matrix:
> 
> $P = \begin{bmatrix} p_{11} & p_{12}\\ p_{21} & p_{22} \end{bmatrix}$
> 
> This matrix is to be understood as follows. For features $x_1$ and $x_2$:
> 
> $p_{ij} = P(x_i = 1\ |\ y = j)$
> 
> In $p_{ij}$, the first index stands for the feature and the second stands for the class-label.
> 
> <HR>
> 
> Write a function named `bernoulli_naive_bayes` that accepts a feature matrix `X` and a label vector `y` as arguments. It should return the parameter matrix `P`. Both the arguments and the return value are of type `np.ndarray`. You can assume that no smoothing is required.
```
import numpy as np
def bernoulli_naive_bayes(X, y):
    """
    Estimate the parameter matrix
    
    Arguments:
    	X: feature matrix, (100, 2), np.ndarray
    	y: label vector, (100, ), np.ndarray
    Return:
    	P: parameter matrix, (2, 2), np.ndarray
    """
    n_samples,n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    w = np.zeros((n_features,n_classes),dtype=np.float64)
    for idx, c in enumerate(classes):
        X_c = X[y==c]
        w[idx,:] = np.sum(X_c,axis = 0)/X_c.shape[0]
    return w.T
```

### PPA - 2
> You are given a numerical data matrix $x$ as an `np.ndarray` shape $(200 \times 5)$ and a vector of class labels $y$ size $(200)$ as `np.ndarray` for a multi-class classification problem. Define a function `mean_estimate` which calculates the estimated mean of data samples corresponding to the class labels for each feature and returns a dictionary with class labels as keys and estimated mean vectors as values. The $i^{th}$ element of a mean vector corresponds to the $i^{th}$ feature.
```
import numpy as np
def mean_estimate(X: np.ndarray,  y : np.ndarray):
    """
    Estimate the mean of samples for each class

    Arguments:
        X: samples, (200, 5), np.ndarray
        y: labels, (200, ), np.ndarray
    Return:
        D: dictionary
            key: label, int
            value: mean, np.ndarray
    """
    n_samples, n_features = X.shape
    class_count = np.unique(y)
    n_classes = len(class_count)
    mean_est = {}
    for idx, c in enumerate(class_count):
        X_c = X[y==c]
        mean_est[c] = X_c.mean(axis = 0)
    return mean_est
```

### PPA - 3
> Write a function naive_gaussian_predict that implements a Gaussian Naive Bayes model on training data, and returns the predicted class labels for the test data. This is a binary classification task (labels are $0$ and $1$) and the size of `X_train`, `y_train` and `X_test` are fixed to $(800 \times 2)$, $(800, )$ and $(200 \times 2)$ respectively. The function signature is as follows:
> 
> Arguments:
> - `X_train`: train samples, $(800, 2)$, np.ndarray 
> - `y_train`: train labels,  $(800, )$, np.ndarray 
> - `X_test`:  test samples, $(200, 2)$, np.ndarray
> 
> Return:  
> - y_pred: test labels, $(200, )$ np.ndarray
```
import numpy as np
def fit(X,y):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    mean = np.zeros((n_classes, n_features), dtype = np.float64)
    var = np.zeros((n_classes, n_features), dtype = np.float64)
    priors = np.zeros(n_classes, dtype=np.float64)
    classes = np.unique(y)

    for idx, c in enumerate(classes):
      X_c = X[y==c]
      mean[idx, :] = X_c.mean(axis = 0)
      var[idx,:] = X_c.var(axis = 0)
      priors[idx] = X_c.shape[0]/ float(n_samples)
    
    return (mean,var,priors,classes)

def calc_pdf(class_idx,X,mean,var):
    mean = mean[class_idx]
    var = np.diag(var[class_idx])
    z = np.power(2*np.pi,X.shape[0]/2) * np.power(np.linalg.det(var),1/2)
    return (1/z) * np.exp(-(1/2)*(X-mean).T @ (np.linalg.inv(var)) @(X-mean))

def calc_prod_likelihood_prior(X,classes,priors,mean,var):
    prod_likelihood_prior = np.zeros((X.shape[0], len(classes)),dtype = np.float64)
    for x_idx, x in enumerate(X):
      for idx, c in enumerate(classes):
        prod_likelihood_prior[x_idx, c] = (np.log(calc_pdf(idx,x,mean,var)) + np.log(priors[idx]))
    
    return prod_likelihood_prior

    
def naive_gaussian_predict(X_train, y_train, X_test):
    """
    Train a Gaussian NB and predict the labels on test set 

    Arguments:
        X_train: train samples, (800, 2), np.ndarray 
        y_train: train labels,  (800, ), np.ndarray 
        X_test:  test samples, (200, 2), np.ndarray
    Return  
        y_pred: test labels, (200, ) np.ndarray
    """
    mean,var,priors,classes = fit(X_train,y_train)
    likelihood_prior = calc_prod_likelihood_prior(X_test,classes,priors,mean,var)
    return np.argmax(likelihood_prior, axis = 1)
```

### PPA - 4
> For a binary classification problem with class labels ($0$ and $1$), define a function `class_scores` that accepts the true and predicted labels and returns the following evaluation metrics as a dictionary.  
> 1. Precision  
> 2. Recall  
> 3. Accuracy  
> 4. F1 score  
> 5. Misclassification Rate
> 
> They keys of the dictionary are the names of the metrics, exactly as they are given above. The values are the corresponding measurements expressed as floats. The function should have the following signature:
> 
> Arguments:  
> - `y_test`: true labels, $(n, )$, np.ndarray 
> - `y_pred`: predicted labels, $(n, )$, np.ndarray
> 
> Return:
> - metrics: dictionary
> - key: string, names of the metrics
> - value: float
> 
> Both `numpy` arrays are of size $(n, )$. Do not use any existing methods/functions to calculate the same. Consider label 1 as the positive class. Note that the misclassification rate is 1 minus the accuracy.
```
import numpy as np
def class_scores(y_test: np.ndarray, y_pred: np.ndarray):
    """
    Compute evaluation metrics for a binary classification task
    
    Arguments:  
        y_test: true labels, (n, ), np.ndarray 
        y_pred: predicted labels, (n, ), np.ndarray
    Return:
        metrics: dictionary
            key: string, names of the metrics
            value: float
    """
    TP = np.where((y_pred== 1) & (y_test == 1),1,0).sum()
    TN = np.where((y_pred== 0) & (y_test == 0),1,0).sum()
    FP = np.where((y_pred== 1) & (y_test == 0),1,0).sum()
    FN = np.where((y_pred== 0) & (y_test == 1),1,0).sum()
    
    classification_report = {}
    
    classification_report["Precision"] = float(TP/(TP+FP))
    classification_report["Recall"] = float(TP/(TP+FN))
    classification_report["Recall"] = float(TP/(TP+FN))
    classification_report["Accurancy"] = float((TP+TN)/(TP+TN+FP+FN))
    classification_report["F1 Score"] = float((2*float(TP/(TP+FP))*float(TP/(TP+FN)))/(float(TP/(TP+FP))+float(TP/(TP+FN))))
    classification_report["Misclassification Rate"] = (1-float((TP+TN)/(TP+TN+FP+FN)))
    
    return classification_report
```

### GrPA - 1
> In a multi-class classification setting, consider a numerical feature matrix $X$ as an  np.ndarray  of shape $(n, m)$ and a vector of class labels  $y$  of size $(n, )$ as an  np.ndarray. Define a function `variance_estimate` which calculates the estimated variance of data samples corresponding to individual class labels for each feature. The function should return a dictionary with class labels as keys and estimated variance vectors as values. The $i^{th}$ element of a vector corresponds to the variance of the $i^{th}$ feature.
```
import numpy as np
def variance_estimate(X: np.ndarray, y: np.ndarray):
    class_count = np.unique(y)
    class_dic = {}
    for c in (class_count):
        X_c = X[y == c]
        class_dic[c] = X_c.var(axis = 0)
    return class_dic
```

### GrPA - 2
> Write a function `naive_gmodel_eval` that implements a Gaussian Naive Bayes model on training data, predicts class labels for the test data and returns the evaluation scores for the predictions performed on the test data, corresponding to  **each individual label**  appearing on `y_test`. This is a multiclass classification task and the size of  `X_train`, `y_train`, `X_test` and `y_test`  are fixed to  $(1800, 5)$, $(1800, )$, $(200,5)$ and $(200,)$  respectively. There are four classes labeled as  $0, 1, 2, 3$.
> 
> ----------
> 
> For each label, the following evaluation metrics have to be generated by treating that label as a positive class and all others as the negative class.  
> 1. Precision
> 2. Recall
> 3. Accuracy
> 4. F1 score
> 5. Misclassification Rate
> 
> This information has to be stored in a dictionary. They keys of the dictionary are the names of the metrics, exactly as they are given above. The values are the corresponding measurements expressed as floats.  
> 
> ----------
> 
> Create a parent dictionary named `metrics`. The keys of this dictionary will be the labels and the values will be the corresponding evaluation dictionaries. So, your function should return  metrics, which is essentially a dictionary of dictionaries!
> 
> ----------
> 
> The function will have the following signature:
> 
> Arguments:
> - X_train: training samples, (1800, 5), np.ndarray
> - y_train: training labels, (1800, ), np.ndarray, labels are 0, 1, 2 or 3
> - X_test:  test samples, (200, 5), np.ndarray
> - y_test:  test_labels, (200, ), np.ndarray, labels are 0, 1, 2, or 3
> 
> Return:
> - metrics: dict of dicts
> - key: label, int
> - value: dict
> - key: string
> - value: float
```
import numpy as np
def fit(X, y):
    n_samples, n_features = X.shape
    class_count = np.unique(y)
    n_classes = len(class_count)
    mean = np.zeros((n_classes, n_features), dtype = np.float64)
    var = np.zeros((n_classes, n_features), dtype = np.float64)
    prior = np.zeros((n_classes), dtype = np.float64)
    for idx, c in enumerate(class_count):
        X_c = X[y == c]
        mean[idx,:] = X_c.mean(axis = 0)
        var[idx,:] = X_c.var(axis = 0)
        prior[idx] = X_c.shape[0] / float(n_samples)
    return (mean, var, prior, class_count)

def calc_pdf(idx, X, mean, var):
    mean = mean[idx]
    varx = np.diag(var[idx])
    z = np.power(2 * np.pi, X.shape[0] / 2) * np.power(np.linalg.det(varx), 0.5)
    return ((1 / z) * np.exp((-1 / 2) * (X - mean).T @ np.linalg.inv(varx) @ (X - mean)))

def log_likelihood_prior(X, classes, mean, var, prior):
    prod_likelihood_prior = np.zeros((X.shape[0], len(classes)), dtype = np.float64)
    for x_idx, x in enumerate(X):
        for idx, c in enumerate(classes):
            prod_likelihood_prior[x_idx, c] = np.log(calc_pdf(idx, x, mean, var)) + np.log(prior[idx])
    return prod_likelihood_prior

def predict(X):
    return np.argmax(X, axis = 1)

def naive_gmodel_eval(X_train, y_train, X_test, y_test):
    (mean, var, prior, class_count) = fit(X_train, y_train)
    Prod_Log_like = log_likelihood_prior(X_test, class_count, mean, var, prior)
    y_pred = np.argmax(Prod_Log_like, axis = 1)
    metrics = {}
    for c in class_count:
        TP = np.where((y_pred == c) & (y_test == c), 1, 0).sum()
        TN = np.where((y_pred != 0) & (y_test != c), 1, 0).sum()
        FP = np.where((y_pred == c) & (y_test != c), 1, 0).sum()
        FN = np.where((y_pred != c) & (y_test == c), 1, 0).sum()
        metrics[c] = {}
        metrics[c]["Precision"] = float(TP / (TP + FP))
        metrics[c]["Recall"] = float(TP / (TP + FN))
        metrics[c]["Accurancy"] = float((TP + TN) / (TP + TN + FP + FN))
        metrics[c]["F1 Score"] = float((2 * float(TP / (TP + FP)) * float(TP / (TP + FN))) / (float(TP / (TP + FP)) + float(TP / (TP + FN))))
        metrics[c]["Misclassification Rate"] = (1 - float((TP + TN) / (TP + TN + FP + FN)))
    return metrics
```

<H1 ALIGN=CENTER> Week - 7 </H1>

### PPA - 1
> Write a function `euclid(a, b)` to find Euclidean distance between vectors `a` and `b`. Both `a` and `b` have shape $(n, 1)$, where $n$ is number of features/dimensions.
> 
> Input:
> - Vectors `a` and `b`.
> 
> Output:
> - Euclidean distance between vectors `a` and `b`.
```
import numpy as np
def euclid(a, b):
    return np.sum((a - b) ** 2, axis = 1)
```

### PPA - 2
> Write a function `one_hot(y)` which performs one hot encoding on vector `y` and then outputs a resultant matrix which can be used for softmax regression as output label matrix. `y` is row matrix with $(n, 1)$ shape, where n is number of samples.
> 
> Example:
> If $y$ is $[8, 6, 3]$, its one hot encoding will be $\begin{bmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 0 & 0 \end{bmatrix}$.
> Input:
> - `y`: A vector of shape $(n, 1)$
> 
> Output:
> - A output label matrix of suitable shape.
```
import numpy as np
def one_hot(y):
    encoded = np.zeros((y.size, y.max() + 1))
    encoded[np.arange(y.size), y] = 1.0
    return encoded
```

### GrPA - 1
> Write a function `manhattan(a, b)` to find Manhattan distance between vectors `a` and `b`. Both `a` and `b` are vectors with shape $(n, 1)$ where $n$ is number of features/dimensions.
> 
> Input:
> - Vectors `a` and `b`.
> 
> Output:
> - Manhattan distance between vectors `a` and `b`.
```
import numpy as np
def manhattan(a,b):
	M=np.sum(np.abs(a-b),axis=0)
	return M
```

### GrPA - 2
> Write a function softmax(Z) to find softmax of linear combination of feature matrix and weight vector. Take care of numerical stability as well.
> 
> Input: Z = X@W, where X is feature matrix of shape (n,m), W is weight matrix (m,k), where n = number of rows, m = number of features and k = number of labels.
> 
> Output: A matrix of (n,1) shape. Each row corresponds to the label of that row. Each label will have value from 0 to k-1.
```
import numpy as np
def softmax(Z):
	m = 1/(1+np.exp(-1))
	return np.round(m)
```

### GrPA - 3
> Write a function knn(class1, class2, x_new) to find in which cluster x_new belongs using 3-NN. Assume cluster 1 and cluster 2 has 5 points each. Use Euclidian distance as distance measure.
> 
> Input:
> - Two numpy arrays class1 and class2 of shape (5,2) each and a numpy array x_new of shape (2,1)
> 
> Output:
> - Return class id to which x_new belongs (i.e. 1 or 2).
```
import numpy as np
from scipy import stats
def knn(class1,class2,x_new):
    x_new=x_new.reshape((1,2))
    x=np.array(list(class1)+list(class2))
    y=np.zeros((10,1))
    y[[0,1,2,3,4]]=1
    y[[5,6,7,8,9]]=2
    # dis_vec= np.sum((x-x_new)**2,axis=1)
    dis_vec=[]
    for i in x:
        dis_vec.append(np.sum((i-x_new)**2))
    dis_vec=np.array(dis_vec)
    k_labels=np.argpartition(dis_vec,3)[:3]
    cluster= stats.mode(y[k_labels])[0]
    return cluster
```


<H1 ALIGN=CENTER> Week - 8 </H1>

### PPA - 1
> Write a function named `hinge_loss` that computes the hinge loss value for corresponding elements of two vectors namely `y_test` $(5 \times 1)$ and `y_pred` $(5 \times 1)$ and returns the mean of the loss values computed.

Note:
`y_test` is the true test label and `y_pred` is the the predicted label.
```
import numpy as np
def hinge_loss(y_pred, y_test):
    return np.mean([max(0, 1 - x * y) for x, y in zip(y_test, y_pred)])
```

### PPA - 2
> Write a function `solve_eqn` to obtain the weight vector (bias as its last element) of linear SVM model by accepting an array of support vectors A of shape $(3 \times 2)$ and their label vector `b` of shape $(3 \times 1)$. This function should return weight vector of shape $(3 \times 1)$.
```
import numpy as np
def solve_eqn(A, b):
    A = np.column_stack((A, np.ones(A.shape[0])))
    S = [[i @ j for j in A] for i in A]
    x = np.linalg.solve(S, b)
    w = np.zeros_like(x)
    for i in range(len(x)):
        w += x[i] * A[i]
    return w
```

### PPA - 3
> Write a class named `fit_softsvm` that implements soft margin SVM using GD. 
> Write a separate function named `support_vectors` and return the `support_vectors` identified by the model. 
> The inputs to the `support_vectors` function should be feature matrix `X_train` and label vector `y_train`.
> 
> Use the following parameters:
> 1.   learning rate = 0.001
> 2.   C = 500
> 3.   epochs = 100
```
import numpy as np
class fit_softSVM:
  def __init__(self,C):
    self._support_vectors = None
    self.C = C
    self.w = None
    self.b = None
    self.X = None
    self.y = None
    #n is the number of data points
    self.n = 0
    #d is the number of dimensions
    self.d = 0
  def __decision_function(self,X):
    return X.dot(self.w) + self.b
  def __cost(self,margin):
    return (1/2)*self.w.dot(self.w) + self.C * np.sum(np.maximum(0,1-margin))
  def __margin(self,X,y):
    return y*self.__decision_function(X)
  def fit(self,X,y,lr=1e-3,epochs=500):
    # Initialize w and b
    self.n, self.d = X.shape
    self.w = np.random.randn(self.d)
    self.b = 0
    #Required only for plotting
    self.X = X
    self.y = y
    loss_array = []
    for _ in range(epochs):
      margin = self.__margin(X,y)
      loss = self.__cost(margin)
      loss_array.append(loss)
      misclassified_pts_idx = np.where(margin<1)[0]
      d_w = self.w - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
      self.w = self.w - lr*d_w
      d_b = -self.C*np.sum(y[misclassified_pts_idx])
      self.b = self.b - lr * d_b 
    self._support_vectors = np.where(self.__margin(X,y)<=1)[0]
  def predict(self,X):
    return np.sign(self.__decision_function(X))
  def score(self,X,y):
    P = self.predict(X)
    return np.mean(y==P)
# Main function
def support_vectors(X_train, y_train):
      svm = fit_softSVM(C=500)
      svm.fit(X_train,y_train)
      return svm._support_vectors
```

### GrPA - 1
> Write a function named `hinge_loss` that computes the hinge loss value for two vectors namely `y_test` $(20 \times 1)$ and `y_pred` $(20 \times 1)$ , and returns the mean of the loss values computed.
> 
> Note:
> `y_test` is the true test label and `y_pred` is the the predicted label.
```
import numpy as np
def hinge_loss(y_pred, y_test):
    return np.sum(y_pred != y_test)
```

### GrPA - 2
> Write a function `fit` to compute gradient of the hinge loss function without regularization. This fit function should have the following input arguments-
> 
> - `X_train`: feature matrix for training
> - `Y_train`: labels for training
> - `X_test`: features for testing
> - `Y_test`: labels for testing
> - `n_iters`: Number of iterations with default value 100
> - `lr`: learning rate with default value 0.1
> 
> The function should return the accuracy using the testing data.
```
import numpy as np

def decision_function(X,w,b):
    return X.dot(w) + b
    
def cost(w,C, margin):
    return (1 / 2) * w.dot(w) + C * np.sum(np.maximum(0, 1 - margin))

def margin(X, y, w, b):
    return y * decision_function(X, w, b)

def fit(X_train, Y_train, X_test, Y_test, n_iters = 100, lr = 0.1):
    n, d = X_train.shape
    w = np.random.rand(d)
    b, C = 0, 1
    X, y = X_train, Y_train
    loss_array = []
    for _ in range(n_iters):
        Margin = margin(X, y, w, b)
        loss = cost(w, C, Margin)
        loss_array.append(loss)
        misclassified_pts_idx = np.where(Margin < 1)[0]
        d_w = w - C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
        w = w - lr * d_w
        d_b =- C * np.sum(y[misclassified_pts_idx])
        b = b - lr * d_b
    support_vectors = np.where(margin(X, y, w, b) <= 1)[0]
    y_hat = np.sign(decision_function(X_test, w, b))
    accuracy = np.mean(Y_test == y_hat)
    return accuracy
```

### GrPA - 3
> Write a class named `fit_softsvm` that implements soft margin SVM using GD.
> Write a separate function named `compute_accuracy` and return the accuracy value using a `pred_accuracy` function which is defined inside the `fit_svm` class.
> 
> The inputs to the `compute_accuracy` function should be feature matrices `X_train`, `X_test` and label vectors `y_train`, `y_test`.
> 
> Use the following parameters:
> 1. learning rate = 0.01
> 2. C = 15
> 3. epochs = 100
```
import numpy as np
class fit_softSVM:
    def __init__(self, C=15):
        self._support_vectors = None
        self.C = C
        self.w, self.b, self.X, self.y = None, None, None, None
        #number of data points
        self.n = 0
        #number of dimensions
        self.d = 0

    def __decision_function(self, X):
        return X.dot(self.w) + self.b
 
    def __cost(self, margin):
        return (1 / 2) * self.w.dot(self.w) + self.C * np.sum(np.maximum(0, 1 - margin))
 
    def __margin(self, X, y):
        return y * self.__decision_function(X)
 
    def fit(self, X, y, lr = 0.01, epochs = 100):
        self.n, self.d = X.shape
        self.w = np.random.rand(self.d)
        self.b = 0
        self.X = X
        self.y = y
        loss_array = []

        for _ in range(epochs):
            margin = self.__margin(X, y)
            loss = self.__cost(margin)
            loss_array.append(loss)
            misclassified_pts_idx = np.where(margin < 1)[0]
            d_w = self.w - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            self.w = self.w - lr * d_w
            d_b = - self.C* np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b

        self._support_vectors = np.where(self.__margin(X,y) <= 1)[0]
 
    def predict(self, X):
        return np.sign(self.__decision_function(X))

    def score(self, X, y):
        P = self.predict(X)
        return np.mean(y == P)

def compute_accuracy(X_train, y_train, X_test,  y_test):
      svm = fit_softSVM(C = 15)
      svm.fit(X_train, y_train)
      return svm.score(X_test, y_test)
```


<H1 ALIGN=CENTER> Week - 9 </H1>

### PPA - 1
> Consider a regression problem with feature matrix $X$ with size $(100 \times 10)$ and label vector y with size $(100 \times 1)$. We split this root node into two nodes `node1` and `node2` according to $j^{th}$ split variable and split value `s`.
> 
> Define a function `predict_node(X, y, j, s)` that takes `X`, `y`, split variable `j` and split value `s` as parameters and returns the tuple of predict values (mean value) at both the nodes.
> 
> - X = ndarray of size (100, 10) with entries as float.
> - y = ndarray of size (100, 1) with entries as float.
> - j = int
> - s = float
```
import numpy as np
def predict_node(X, y, j, s):
    node1 =[]
    node2 =[]
    for i in range((X.shape[0])):
        if X[i][j]<=s:
            node1.append(y[i])
        else:
            node2.append(y[i])
    m1= sum(node1)/len(node1)
    m2= sum(node2)/len(node2)
    return ((m1[0], m2[0]))
```

### PPA - 2
> Define a function `predict_class(y)` that takes the parameter
> 
> `y` which is a ndarray of actual outputs of all the samples in a particular node and retruns the predict class for the same node.
> 
> Note:
> - number of classes are $10$ $(0$ $to$ $9)$.
> - If two classes have same number of samples, function should return the lower class.
```
import numpy as np
def predict_class(y):
    idx={}
    for i in y:
        if not i in idx:
            idx[i]=1
        else:
            idx[i]+=1
    key= list(idx.keys())
    val= list(idx.values())
    for i in range(len(val)-1):
        index=i
        for j in range(i+1, len(val)):
            if val[j]<val[index]:
                index=j
            elif val[j]==val[index]:
                if key[j]<key[index]:
                    index=j
        val[i], val[index]= val[index], val[i]
        key[i], key[index]= key[index], key[i]
    max1= val[-1]
    index= key[-1]
    for i in range(len(val)-2,-1,-1):
        if val[i]==max1:
            index= key[i]
        else:
            break
    return index
```

### PPA - 3
> Define a function `misclassification_error(y)` that takes the parameter `y` which is a ndarray of actual outputs of all the samples in a particular node and return the misclassification error of the same node.
```
import numpy as np
import random
def misclassification_error(y):
    idx={}
    for i in y:
        if not i in idx:
            idx[i]=1
        else:
            idx[i]+=1
    key= list(idx.keys())
    val= list(idx.values())
    for i in range(len(val)-1):
        index=i
        for j in range(i+1, len(val)):
            if val[j]>val[index]:
                index=j
            
        val[i], val[index]= val[index], val[i]
        key[i], key[index]= key[index], key[i]
    max_class=key[0]
    mis_count=0
    for i in y:
        if i != max_class:
            mis_count += 1
    return (mis_count/len(y))
```

### GrPA - 1
> Define a function `gini_index(dict)` having the following characteristics:
> 
> Input:
> - `dict` = dictionary that has classes as keys and 'number of samples in respective class' as values of a particular node.
> 
> Output:
> - Gini index of the same node (a float value).
```
import numpy as np
def gini_index(dict):
    classes = dict.keys()
    total_samples = sum(dict.values())
    Gini = 0
    for k in classes:
        p = dict[k] / total_samples
        Gini += p * (1-p)
    return Gini
```

### GrPA - 2
> Define a function `entropy(dict)` having the following characteristics:
> 
> Input:
> - `dict` = dictionary that has classes as keys and 'number of samples in respective class' as values of a particular node.
> 
> Output:
> - Entropy of the same node. (A float value)
```
import numpy as np
def entropy(dict):
    classes = dict.keys()
    samples = sum(dict.values())
    Entropy = 0
    for k in classes:
        p = dict[k] / samples
        Entropy += (-p * np.log2(p))
    return Entropy
```

### GrPA - 3
> Consider a regression problem using CART. If `y` is a ndarray of targets of the samples in a particular node, define a function `sseloss(y)` that returns the error associated with that node.
```
import numpy as np
def sseloss(y):
    y_mean = np.mean(y)
    error = np.sum((y - y_mean) ** 2, axis = 0)
    return error
```


<H1 ALIGN=CENTER> Week - 10 </H1>

### PPA - 1
> Write a function `similarity(res, lambda_)` which takes residuals at a node as input and returns similarity score which can be used for XGBoost regression.  
> 
> Input:
> - A numpy array having residuals at a node.
> - lambda is the regularization rate.
> 
> Output:
> - Similarity score of the node.  
> 
> Note:
> Similarity Score = (Sum of residuals) 2 / (No. of residuals + lambda_)
```
import numpy as np
def similarity(res,lambda_):
    num = res.shape[0]
    SS = (np.sum(res))**2/(num+lambda_)
    return SS
```

### PPA - 2
> Write a function `gradboost(model,X_train, y_train, X_test, boosting_rounds, learning_rate)` to implement Gradient boost algorithm.  
> 
> Input:  
> - `model`: model to be fitted
> - `X_train`: Training features
> - `y_train`: Training output labels
> - `X_test`: Test feature values
> - `boosting_rounds`: number of boosting rounds
> - `learning_rate`: learning rate used in the algorithm
> 
> Output:  
> - `y_hat_train`: Prediction on training data after number of boosting rounds  
> - `y_hat_test`: Prediction on testing data after number of boosting rounds
```
import numpy as np
import pandas as pd
def gradboost(model, X_train, y_train, X_test, boosting_rounds, learning_rate):
    y_hat_train= np.repeat(np.mean(y_train), len(y_train))
    y_hat_test= np.repeat(np.mean(y_train), len(X_test))
    res= y_train-y_hat_train
    
    for i in range(0, boosting_rounds):
        model= model.fit(X_train,res)
        y_hat_train = y_hat_train+ learning_rate* model.predict(X_train)
        y_hat_test = y_hat_test+ learning_rate* model.predict(X_test)
        res= y_train-y_hat_train
    return y_hat_train, y_hat_test
```

### GrPA - 1
> Write a function `residual(y)` which takes a numpy array `y` as input and calculates residuals for base model (taking mean of the target values).
> 
> Input:
> - numpy array `y` having all the target values.  
> 
> Output:
> - numpy array containing residuals.
```
import numpy as np
def residual(y):
    y_hat = np.repeat(np.mean(y), len(y))
    res= y - y_hat
    return res
```

### GrPA - 2
> Write a function `accuracy(y_true, y_pred)` which calculates accuracy of classification based on inputs `y_true` and `y_pred`.
> 
> Input:
> - `y_true`: vector of true output labels
> - `y_pred`: vector of predicted output labels.  
> 
> Output:
> - accuracy of classification
```
import numpy as np
def accuracy(y_true,y_pred):
    acc= np.sum(y_true==y_pred)/len(y_true)
    return acc
```

### GrPA - 3
> Write a function `bag(X, y)` for creating q bootstrap from original dataset.
> 
> Input:
> - `X` is the feature matrix and `y` is the output label vector  
> 
> Output:
> - Bootstrap containing two numpy arrays containing feature values and corresponding target values.
```
import numpy as np
def bag(X, y):
    #D is no. of samples
    D = X.shape[0]
    np.random.seed(42)
    indices = np.random.choice(range(D), D, replace = True)
    return X[indices], y[indices]
```


<H1 ALIGN=CENTER> Week - 11 </H1>

### PPA - 1
> Write a function euclid(a,b) to find Euclidean distance between vectors a and b.
> 
> Input:
> - Vectors `a` and `b`.
> 
> Return:
> - Euclidean distance between vectors `a` and `b`
> 
> Note:
> - Both vectors are of shape $(m, )$, where $m$ is some positive integer.
```
import numpy as np
def euclid(a, b):
    return np.linalg.norm(a - b)
```

### GrPA - 1
> Write a function `centroid(a, b)` to find centroid of vectors `a` and `b`.
> 
> Input:
> - `a` and `b` are numpy arrays of same shape.
> 
> Output:
> - centroid of `a` and `b` as numpy array.
```
import numpy as np
def centroid(a, b):
    c=[]
    count = a.shape[0]
    for i in range(count):
        c.append((a[i] + b[i]) / 2)
    return np.array(c)
```

### GrPA - 2
> Write a function `silhoutte(a, b)` to calculate silhoutte coefficient.  
> 
> Input:
> - `a` is the mean distance between the instances in the cluster
> - `b` is the mean distance between the instance and the instances in the next closest cluster.  
> 
> Output:
> - Silhoutte coefficient
```
import random
def silhoutte(a, b):
    s = a * b / max(a, b)
    return s
```


<H1 ALIGN=CENTER> Week - 12 </H1>

### PPA - 1
> Write a function named `relu` that accepts a matrix $Z$ as argument and returns the result of the ReLU activation function applied on $Z$ element-wise.
```
import numpy as np
def relu(Z):
    Re = np.where(Z >= 0, Z, 0)
    return Re
```

### PPA - 2
> Write a function named `count_params` that accepts a list  layers  as argument. The first and last element of  layers  corresponds to the input and output neurons. The rest of the elements are the number of neurons in the hidden layers. The function should return the total number of parameters $(weights + biases)$ in the network.
```
import numpy as np
def count_params(layers):
    count = 0
    for i in range(1, len(layers)):
        weights = layers[i - 1] * layers[i]
        bias = layers[i]
        count += (weights + bias)
    return count
```

### PPA - 3
> Consider a multi-class classification setup with  $k$  classes. The labels are integers in the range  $[0, k - 1]$. $Z$ is a matrix of pre-activations of shape  $(n \times k)$ at the output layer of a neural network. $n$ is the batch-size here.
> 
> ----------
> 
> Write a function named `predict` that accepts the matrix $Z$ as argument. Within the function perform the following operations:
> 1. apply the Softmax non-linear function row-wise on $Z$.
> 2. compute the vector of labels of size $(n, )$ predicted by the network
> 
> The function should return this vector.
```
import numpy as np
def softmax(Z):
    expZ = np.exp(Z)
    prob= expZ / expZ.sum(axis = 1, keepdims = True)
    return prob
def predict(Z):
    y_hat = softmax(Z)
    if Z.shape[-1] == 1:
        return y_hat
    else:
        return np.argmax(y_hat, axis = 1)
```

### GrPA - 1
> Write a function named `forward_layer` that accepts three arguments:
> 1. `A`: input activations to layer $l$.
> 2. `W`: weight matrix at layer $l$.
> 3. `b`: bias vector at layer $l$.
> 
> Return the output activation matrix at layer $l$ using Sigmoid activation function.
```
import numpy as np
def forward_layer(A, W, b):
    Z = (A @ W) + b
    A_out= 1 / (1 + np.exp(-Z))
    return A_out
```

### GrPA - 2
> $A^{(g)}$ or `grad_A` is the gradient of the loss with respect to the activations at a hidden layer in a neural network.  ${Z^{(g)}}$ or `grad_Z` is the gradient of the loss with respect to the pre-activations at this layer.  $Z$ or `Z` is a matrix of pre-activations at this layer. The activation function for this layer is ReLU.
> 
> Write a function named `grad_estimate` that accepts `grad_A` and `Z` as arguments and returns `grad_Z`.
```
import numpy as np
def grad_relu(Z):
    return np.where(Z >= 0, 1, 0)
def grad_estimate(grad_A, Z):
    grad_Z = grad_A * grad_relu(Z)
    return grad_Z
```
