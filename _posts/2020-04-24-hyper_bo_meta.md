---
title: 'Hyperparameter Configuration using Bayesian Optimization and Improvements with Meta-Learning'
date: 2020-04-24
permalink: /posts/2020/04/hyper_bo_meta/
tags:
  - bayesian_optimization
  - meta_learning
---

This is a blog post credit to Yongkyun Lee, Emily Park, Alex Pan, Kevin Huang, and Megan Tjandrasuwita

Most algorithms designed for solving complex problems, such as a learning algorithm, have some configurable parameters, i.e. hyperparameters. The value of these parameters can often have a large impact on how well the algorithm performs, which makes it important to have an efficient method to select an optimal set of hyperparameter values given any algorithm and an objective function to measure its performance.

Traditionally, this is done by manually picking reasonable values, or through some brute force method such as a grid search, where all possible combinations of parameter values within defined ranges are tested. However, evaluating the objective function of an algorithm given a parameterization is often very expensive; for example, if our algorithm was deep neural network model, for every parameterization we wanted to test, we would need to train the network, which can be very computationally expensive. In this blog post, we will explore Bayesian optimization, a common efficient method that is applied to configuring hyperparameters, and ways to improve it using meta-learning.

## Introduction

### Bayesian Optimization

Finding a good set of hyperparameters for a machine learning model is an important but time-consuming problem. [Bayesian optimization](https://docs.google.com/presentation/d/1Vgzdicbgumy2uBYrUvoPyQ-CVo8Gus_Iysli3Hv_ulU/edit#slide=id.p) is a general technique of optimizing an unknown computable function $f$ by estimating a probability distribution of $f$, which is then updated based on evaluations of that function. We start off with a prior on $f$, and then given a set of points $\mathbf{x}$ on which we evaluate $f$, we calculate updated beliefs on $f$ using Bayesian inference:

$$P(f | \mathbf{x}) = \frac{P(\mathbf{x} | f) \cdot P(f)}{P(\mathbf{x})}$$

This process can then be repeated by evaluating more points to obtain a more accurate model of $f$. An implementation of Bayesian optimization would involve a model to use for the prior, and at every step, determining points at which to evaluate $f$. 

### SMBO: Testing Hyperparameters Sequentially

The most common formalization of Bayesian optimization for hyperparameter selection is known as sequential model-based optimization, or SMBO. The function to optimize is the objective function $f$ by which we measure our algorithm performance given a particular dataset $\mathcal{D}$, as a function of the hyperparameters $\theta$, i.e the optimization problem is $\displaystyle \text{arg}\min_{\theta} f^{\mathcal{D}}(\theta)$. A particular model of the probability distribution of $f$ must be defined, along with a prior. A common choice is a Gaussian process, which models each point as a Gaussian distribution. At each step, SMBO then picks the best point at which to compute $f$ and updates the probability model, which is determined by optimizing some selection or acquisition function. The most common choice is expected improvement, which measures the expected value of the improvement over the current optimum for every point. SMBO terminates with some stopping condition and returns the optimal hyperparameters found. 

Let's formalize the problems we are tackling in mathematical terms.

SMBO takes in $\mathbf{\Theta} = \Theta_1 \times  … \times \Theta_n$, which is the space of hyperparameters $\theta_1,  …, \theta_n$. In other words, $\Theta$ consists of all possible combinations of hyperparameter values. The objective function to optimize over for a given dataset $\mathcal{D}$ is the algorithm’s validation error,
$$f^{\mathcal{D}}(\theta) = \dfrac{1}{k}\sum_{i=1}^{k}\nu(\theta, \mathcal{D}^{(i)}_{train}, \mathcal{D}^{(i)}_{valid})$$
 when it is trained with a set of hyperparameters $\theta$. Starting by evaluating a list of initial configurations $\mathbf{\theta}_{1:t}$, we proceed with iterations of SMBO until time limit $T$ is reached, as described in the below [pseudocode](http://ceur-ws.org/Vol-1201/paper-03.pdf): 

> Input: Target function $f^{\mathcal{D}}$, limit $T$, hyperparameter space $\Theta$, initial hyperparameter values $\theta_{1:t}=(\theta_{1}, ..., \theta_{t})$
> Output: Best hyperparameter configuration $\theta^{*}$ found
> for $i = 1$ to $t$, do $y_{i} \leftarrow$ evaluate $f^{D}(\theta_{i})$.
> for $j = t+1$ to $T$, do
> &nbsp;&nbsp; $M \leftarrow$ fit model on performance data $<\theta_{i}, y_{i}>_{i=1}^{j-1}$
> &nbsp;&nbsp; select $\theta_{j} \in argmax_{\theta \in \Theta}a(\theta, M)$
> &nbsp;&nbsp; $y_{j} \leftarrow$ evaluate $f^{D}(\theta_{j})$
> return $\theta^{*} \in argmax_{\theta_{j} \in \{\theta_{1}, ..., \theta_{T} \} } y_{j}$
> 
To explain in words, first we fit a surrogate probabilistic model $M$ to the (input, output) pairs gathered until now. Then, we use $M$ to select a promising input $\theta$ to evaluate next by calculating the acquisition function $a(\theta, M)$, which determines the desirability of the input. We finally evaluate the function at the new input $\theta$. 

In the algorithm, there are three choices that we can make: 1) probabilistic model $M$, 2) acquisition function $a(\theta, M)$, and 3) initialization $(\theta_{1}, ..., \theta_{t})$. For the probabilistic model, popular choices include Gaussian processes and tree-based approaches, like SMAC. For the acquisition function, expected improvement (EI) over the best input at the current state is widely used. These two choices tend to perform well in many types of problems. The initialization process is not as straightforward because there is no set formula to generate a good initialization.

Though SMBO typically is more effective at determining good hyperparameters than traditional methods, it does not retain information across repeated runs. That is, for every dataset $\mathcal{D}$, we need to run a fresh instance of SMBO. Next, we explore the possibility of using meta-learning in combination with the generic algorithm to improve performance.

### Meta-learning and MI-SMBO

Meta-learning is commonly known as _learning to learn_. The goal is to design models that can learn to adapt to new environments quickly using a small training dataset. Among the different types of meta-learning, one type tries to optimize the hyperparameters to improve a target task. A method of choosing initial hyperparmeters is known as Meta-learning-based Initialization (MI-SMBO), which is what we will discuss in this post.

Meta-learning is a quickly growing field of its own, as the rapid learning within tasks and the gradual meta-learning across different tasks, partly due its importance as a building block in [artificial intelligence](https://arxiv.org/abs/1604.00289). The idea of multi-task learning has a long history, and traditional methods have been to directly [train base learners](https://arxiv.org/abs/1605.06065).  Such methods rapidly bind new information after a single presentation while having the ability to slowly learn a method for obtaining useful representations of the data and thus support robust meta-learning.

However, we can still improve on the already-improved method of learning; more recent work has been done to train the algorithms themselves, ie. [learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474). Instead of using hand-designed update rules, learning to learn by gradient descent by gradient descent uses update rules that are learned such that the optimizer is provided with the performance of the optimizee and proposes updates intended to increase this performance, essentially forming a feedback loop that can teach itself. Thus, this method is capable of a more general setting of optimization versus a more supervised learning approach.

With the work being done to advance meta-learning itself, work on MI-SMBO incorporates the advantages and growing improvements of meta-learning with Bayesian optimization. MI-SMBO applies meta-learning to choosing the initial design of SMBO, a choice that does not have a straightforward answer. 

In the below pseudocode, [SMBO with meta-learning initialization](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/10029/9349), MI-SMBO($\mathcal{D}_{new}, f^{\mathcal{D}_{new}}, \mathcal{D}_{1:N}, \hat{\theta}^{1:N}, d, t, T, \Theta$), finds the initial design $\hat{\mathbf{\theta}}_{1:t}$ after given datasets $\mathcal{D}^1, …, \mathcal{D}^N$ and the corresponding best configurations on these previous datasets:


> Input: new dataset $\mathcal{D}_{new}$, target function $f^{D_{new}}$, training datasets $\mathcal{D}_{1:N} = (D_{1}, ... ,D_{N})$, best configurations for training datasets $\hat{\theta}_{1:N} = \hat{\theta_{1}}, ... , \hat{\theta_{N}}$, distance metric $d$, number of configurations to include in initial design $t$, limit $T$, hyperparameter space $\Theta$
> Result: Best hyperparameter configuration $\theta^{*}$ found
> Sort dataset indices $\pi(1), ... , \pi(N)$ by increasing distance to $\mathcal{D}_{new}$, i.e.: $(\pi(i) ≤ \pi(j)) \Leftrightarrow (d(\mathcal{D}_{new}, \mathcal{D}i_{i}) \leq d(\mathcal{D}_{new}, \mathcal{D}_{j}))$
> for $i = 1$ to $t$ do $\theta_{i} \leftarrow \hat{\theta}^{\pi(i)}$
> $\theta^{*} \leftarrow SMBO(f^{\mathcal{D}}, T, \Theta, \theta_{1:t})$
> return $\theta^{*}$

We begin the search for hyperparameter with the configurations that were optimal for similar datasets. In essence, we relate the new dataset to the previous datasets that we experienced. 

The best hyperparameters $\hat{\theta}_{1}, ... , \hat{\theta}_{N}$ for previously experienced datasets $\mathcal{D}_{1}, ..., \mathcal{D}_{N}$ originate from arbitrary sources, including manual search or applications of SMBO to those datasets. $\pi$ denotes a permutation of $(1, ..., N)$ sorted by increasing distance. $d$ is a distance metric between datasets and is defined as $d(\mathcal{D}_{new}, d_{j}) = ||m^{new} - m^{j}||$ where $m^{i} = (m_{1}^{i}, ..., m_{F}^{i})$ is a set of metafeatures set $F$ for datasets.

Metafeatures include simple metafeatures (number of classes, number of features, etc.), information-theoretic metafeatures (class entropy), statistical metafeatures (kurtosis, skewness, etc.), PCA metafeatures (pca 50%, etc.), and landmarking metafeatures (Decision Tree, Naive Bayes, etc.). The below image shows examples of various metafeatures.

<a href="https://www.researchgate.net/figure/List-of-implemented-metafeatures_tbl1_275966998"><img src="https://www.researchgate.net/profile/Joaquin_Vanschoren/publication/275966998/figure/tbl1/AS:613856632139814@1523366234986/List-of-implemented-metafeatures.png" alt="List of implemented metafeatures" width="400"/></a>

Finally, a regressor is used to learn an estimate for a distance metric $d$ between the previous datasets using the chosen metafeatures. Such an estimate is then used to order $\mathcal{D}_1, …, \mathcal{D}_N$ such that the first $t$ configurations for these datasets are chosen as the initial design. 

### Ranking-weighted Gaussian Process Ensemble

One problem of the previous approach is that we must manually define the metafeatures. Also, using past optimization tasks to learn does not scale well with the number of past runs. Instead, we might imagine an approach where we can use a machine learning model to learn the similarities between datasets. That is, we can train a model that predicts a probability distribution of the hyperparameter landscape of a novel dataset, trained on previous optimization runs on other datasets. It should be scalable and does not require metafeatures. One approach taken by [Feurer *et al*. [2018]](https://arxiv.org/pdf/1802.02219.pdf) is to create ranking-weighted Gaussian process ensemble (RGPE). That is, say that we have already performed $t - 1$ optimization runs using a Bayesian optimization process on $t - 1$ different datasets, each using a Gaussian process model, and that we are currently trying to optimize run $t$. Let $\mathcal{D}_i$ denote the set of points for which we have made evaluations for run $i$. That means we have calculated probability distributions $P(f_i | \mathcal{D}_i)$ for $i < t$ and we are currently modelling $P(f_t | \mathcal{D}_t)$. We wish to create a model for $f_t$ that incorporates all of the past $t - 1$ runs; that is, we wish to model the probability distribution $P(f_t | \mathcal{D})$ where $\mathcal{D} = \{\mathcal{D_i},...\mathcal{D_t}\}$. We define our model as an ensemble Gaussian process given by:
$$
P(f_t | \mathcal{D}) = \displaystyle \sum_{i = 1}^{t} w_i P(f_i | \mathcal{D}_i)
$$
This means that
$$
P(f_t | \mathcal{D}) \sim \mathcal{N}\left( \displaystyle \sum_{i = 1}^{t} w_i \mu_i, \displaystyle \sum_{i = 1}^{t} w_i \sigma^2_i\right)
$$
where $\mu_i$ and $\sigma^2_i$ are the mean and variance of each of the individual Gaussian processes. That is, our model for the current target function is a linear combination of our models from our past runs, and the Gaussian process for the current target function itself. We define our loss function as a random variable that is equal to the number of points in $\mathcal{D_t}$ that is misranked by our model. That is:
$$
\mathcal{L}(f, \mathcal{D}_t) = \displaystyle \sum_{j=1}\sum_{k=1}1[f(x_j) < f(x_k)] \oplus (y_j < y_k)
$$
Again, note that the loss is a random variable, so an instance of t he loss requires sampling from $P(f | \mathcal{D})$. Note that for the loss $f_t$ in particular, we must leave out a point from $\mathcal{D}_t$ so that it accurately reflects generalization. To calculate the weights, for every model $i$, we draw $S$ samples from $\mathcal{L}(f_i, \mathcal{D}_t)$: $l_{i, s} \sim \mathcal{L}(f_i, \mathcal{D}_t)$, and set our weight as follows:
$$
w_i = \frac{1}{S} \sum_{s=1}^{S} 1[i = arg\min_{i'}l_{i',s}]
$$
The intuition here is that the weight of each model is the probability that that model has the lowest loss as defined by our loss function. This model provides a good starting point, from which we can improve it by sampling more data points in the same fashion as SMBO. That is, we can use an acquisition function, such as expected improvement, to select another candidate for evaluation, which we add to $\mathcal{D}_t$. We then update our model using the process described above. 

Experimental results from [Feurer *et al*. [2018]](https://arxiv.org/pdf/1802.02219.pdf) have demonstrated that this method converges quicker than a fresh Bayesian optimization run, and this method does not require manually defining any metafeatures. 

The improvements made by meta-learning on Bayesian optimization are similar to improvements we see with other solvers. One such case is creating a SAT solver. It is wildly known that different solvers perform better on different instances. However, the traditional approach of choosing a solver for a class of instances may not always give the best results; the improved approach described by [SATzilla](https://arxiv.org/abs/1111.2249), which constructs portfolios of the best solver on a per-instance basis instead of a per-class basis to make its decisions, is much more effective in terms of speed and robustness. The strategy behind meta-learning fits with what we see with SATzilla, thus complementing meta-learning's effectiveness as well.

## Examples: Comparison of Grid Search, Random Search, and Bayesian Optimization for XGBoost

Using Bayesian optimization instead of grid search or random search has been shown to be [more effective](https://app.sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf), since every time $f$ is evaluated, the algorithm obtains a better idea of the probability distribution of $f$, and thus information is kept which allows it to evaluate $f$ at points more likely to be the optimum. Thus, we can expect to compute $f$ much fewer times compared to something like grid search or random search, which keeps track of no information. Let's see this with code.

We train a XGBoost regressor on sklearn’s toy diabetes dataset in order to predict the response variable, a measure of disease progression, from ten measured baseline variables. The [hyperparameters for XGBoost](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters) include learning rate, gamma, max depth of an estimator, the number of estimators, and the minimum child weight of a leaf node. Learning rate shrinks feature weights while both gamma and minimum child weight, so larger values of these parameters prevent overfitting. On the other hand, increasing maximum depth allows estimators to become more complex, and the complexity of the model also increases with a greater number of estimators.

```python
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score

from scipy.stats import uniform
from xgboost import XGBRegressor

# Load the diabetes dataset (for regression)
X, Y = datasets.load_diabetes(return_X_y=True)
xgb = XGBRegressor()

# Compute a baseline to beat with hyperparameter optimization 
baseline = cross_val_score(xgb, X, Y, scoring='neg_mean_squared_error').mean()
```
We then compare three approaches of hyperparameter tuning: grid search, random search, and Bayesian optimization. Both grid search and random search are implemented in sklearn. The former is an exhaustive search over the hyperparameter space, which must be specified as discrete sets of possible values for each variable, whereas the latter takes in either probability distributions for continuous variables or a range of potential values. Each iteration of random search tests a set of hyperparameter choices, where values for all hyperparameters are sampled independently of each other. 

```python
# Hyperparameters to tune and their ranges for random search
param_dist = {"learning_rate": uniform(0, 1),
              "gamma": uniform(0, 5),
              "max_depth": range(1,50),
              "n_estimators": range(1,300),
              "min_child_weight": range(1,10)}
rs = RandomizedSearchCV(xgb, param_distributions=param_dist, 
                        scoring='neg_mean_squared_error', n_iter=50)
# Run random search for 25 iterations
rs.fit(X, Y);

# Exhaustive grid search
param_dist = {"learning_rate": np.linspace(0, 1, num = 2),
              "gamma": range(0, 5, 2),
              "max_depth": range(20,50,20),
              "n_estimators": range(100,300,100),
              "min_child_weight": range(1,10,4)}
gs = GridSearchCV(xgb, param_grid=param_dist, 
                        scoring='neg_mean_squared_error')
gs.fit(X, Y);
```
Finally, we use [GPyOpt’s](https://gpyopt.readthedocs.io/en/latest/GPyOpt.methods.html) implementation of Bayesian Optimization. The optimization objective function is defined as the validation score over the domain of continuous or discrete hyperparameter values, and we select expected improvement as the acquisition function. A higher value of “jitter” makes the algorithm more explorative, and we make sure that the value is on the same order as our loss function. 

```python
import GPyOpt
from GPyOpt.methods import BayesianOptimization

bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
        {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}]

# Optimization objective 
def cv_score(parameters):
    parameters = parameters[0]
    score = cross_val_score(
                XGBRegressor(learning_rate=parameters[0],
                              gamma=int(parameters[1]),
                              max_depth=int(parameters[2]),
                              n_estimators=int(parameters[3]),
                              min_child_weight = parameters[4]), 
                X, Y, scoring='neg_mean_squared_error').mean()
    score = np.array(score)
    return score

optimizer = BayesianOptimization(f=cv_score, 
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True, 
                                 maximize=True)

# Only 20 iterations because we have 5 initial random points
optimizer.run_optimization(max_iter=50)
```

As our problem is regression, we use negative mean squared error, where a less negative loss is desirable, to score all three models. For grid search, only 2-3 values were considered for each hyperparameter such that it had a runtime on the same scale (less than a minute to complete) as that of the other two methods.

```python
y_rs = np.maximum.accumulate(rs.cv_results_['mean_test_score'])
y_gs = np.maximum.accumulate(gs.cv_results_['mean_test_score'])
y_bo = np.maximum.accumulate(-optimizer.Y).ravel()

print(f'Baseline neg. MSE = {baseline:.2f}')
print(f'Random search neg. MSE = {y_rs[-1]:.2f}')
print(f'Grid Search neg. MSE = {y_gs[-1]:.2f}')
print(f'Bayesian optimization neg. MSE = {y_bo[-1]:.2f}')

plt.plot(y_rs, 'ro-', label='Random search')
plt.plot(y_bo, 'bo-', label='Bayesian optimization')
plt.xlabel('Iteration')
plt.ylabel('Neg. MSE')
plt.ylim(-5000, -3000)
plt.title('Value of the best sampled CV score');
plt.legend();
```

<a href="https://i.imgur.com/7CNR9Lf.png"><img src="https://i.imgur.com/7CNR9Lf.png" alt="Value of the best sampled CV score (MSE)"/></a>


Bayesian optimization achieves the lowest mean squared error on most iterations compared to random search, random search performed slightly worse than the baseline, and grid search performed the worst by far.

### Running the Comparison on Multi-Layered Perceptrons

We train a MLP with two hidden layers for classification on sklearn’s interleaving half circles dataset. This time, the hyperparameters consist only of alpha, the L2 regularization term, and the number of nodes in the two hidden layers. As the dataset only consists of 35 samples, we set the optimizer to “lgbs,” which is a better fit for smaller datasets. Unlike stochastic gradient descent, it does not use minibatches or set the learning rate. 

```python
from sklearn.neural_network import MLPClassifier

# Load the interleaved moons for classification
X, Y = datasets.make_moons(n_samples = 35)

# Instantiate a MLPClassifier with default hyperparameter settings
mlp = MLPClassifier(solver='lbfgs')
baseline = cross_val_score(mlp, X, Y, scoring='neg_log_loss').mean()

# Hyperparameters to tune and their ranges
param_dist = {"alpha": uniform(0, 0.0004),
              "hidden_layer_sizes": (range(1,150), range(1,5))}
rs = RandomizedSearchCV(mlp, param_distributions=param_dist, 
                        scoring='neg_log_loss', n_iter=25)

# Run random search for 25 iterations
rs.fit(X, Y);
```


```python 
bds = [{'name': 'alpha', 'type': 'continuous', 'domain': (0, 0.0004)},
        {'name': 'hidden_layer1_size', 'type': 'discrete', 'domain': range(1, 150)},
        {'name': 'hidden_layer2_size', 'type': 'discrete', 'domain': range(1, 5)}]

# Optimization objective 
def cv_score(parameters):
    parameters = parameters[0]
    score = cross_val_score(
                MLPClassifier(solver='lbfgs',
                              alpha=parameters[0],
                              hidden_layer_sizes=(int(parameters[1]),int(parameters[2]))), 
                X, Y, scoring='neg_log_loss').mean()
    score = np.array(score)
    return score

optimizer = BayesianOptimization(f=cv_score, 
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 10e-4,
                                 exact_feval=True, 
                                 maximize=True)

optimizer.run_optimization(max_iter=20)
```

<a href="https://i.imgur.com/Cf3dKdh.png"><img src="https://i.imgur.com/Cf3dKdh.png" alt="Value of the best sampled CV score (log loss)"/></a>

Again, Bayesian optimization achieves the lowest loss, whereas random search is outperformed by the baseline. In conclusion, SMBO performs better than grid search and random search in the above regression and classification problems, which agrees with the results of [formal studies done on the subject.](http://ceur-ws.org/Vol-1201/paper-03.pdf) 

## Further extensions and final remarks

Optimally tuning hyper-parameters is crucial for model training and leads to great leaps in performance and efficiency. Because of its importance, the problem of tuning hyper-parameters, also known as model configuration, has been well studied. We saw several examples of how developing a model-based approach to algorithm configuration yielded great improvements in both the efficiency and accuracy of models. For high-dimensional spaces, Bayesian optimization emerged as [an efficient approach](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a) compared to a naive grid search algorithm to tune hyper-parameters. Continuing the model-based paradigm, researchers formulated Sequential Model Based Optimization (SMBO) and other iterative approaches to Bayesian optimization. Finally, we applied the idea of meta-learning to SMBO, training a model to distinguish between different problem classes. 

Although we have touted the power of SMBO, let's step back and examine a few of its flaws. In particular, SMBO has [three shortcomings](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) that prevent it from being used in general algorithm configuration: (1) it only allows numerical hyper-parameters, (2) it is unable to terminate operation on poorly performing runs, (3) and it only train algorithms for specific problem instances.

We touched on meta-learning to initialize the parameters of SMBO; this allows SMBO to perform well over collection of problems, instead of one instance. We will now briefly touch on two other approaches to algorithm configuration that address the other two problems. 

Hutter et al. proposed Sequential Model-based Algorithm Configuration ([SMAC]([https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf))) as an extension of SMBO that allows for both categorical and numerical hyper-parameters. Several standard SMBO algorithms rely on Gaussian process models for configuring hyper-parameters because of their desirable features. The standard GP kernel is described as $k : \mathbf{\Theta} \times \mathbf{\Theta} \mapsto \mathbb{R}^+$ with 

$$k(\mathbf{\theta}^1, \mathbf{\theta}^2) = \text{exp}\left[\sum_{l=1}^d\left(-\lambda_l\cdot(\theta^1_l - \theta^2_l)^2\right)\right],$$ 

where $\mathbf{\Theta}$ is the $d$ dimensional hyper-parameter space and $\lambda_1, \ldots,\lambda_d$ are the kernel parameters. To extend this to categorical hyper-parameters, Hutter et al. suggested using a weighted Hamming distance kernel, e.g. 

$$k_{\text{categorical}}(\mathbf{\theta}^1, \mathbf{\theta}^2) = \text{exp}\left[\sum_{l=1}^d \left(-\lambda_l \cdot (1 - \delta(\mathbf{\theta}^1_l, \mathbf{\theta}^2_l)\right)\right],$$ 

where $\delta$ is the Kronecker delta. If the hyper-parameters are a combination of numerical and categorical, one can combine the two kernels as 

$$k_\text{combined}(\mathbf{\theta}^1, \mathbf{\theta}^2) = \text{exp}\left[\sum_{l=1}^d\left(-\lambda_l\cdot(\theta^1_l - \theta^2_l)^2\right) + \sum_{l=1}^d \left(-\lambda_l \cdot (1 - \delta(\mathbf{\theta}^1_l, \mathbf{\theta}^2_l)\right)\right].$$ 

This extension to categorical hyper-parameters provides SMAC more ability to describe problems through their hyper-parameters, leading to a richer understanding of problem classes.

Li et al. developed [Hyperband](https://arxiv.org/abs/1603.06560) to create a hyper-parameter configuration algorithm that stops poorly performing runs. The algorithm disregards Bayesian optimization completely, and instead attempts to tune hyper-parameters through random search. A naive random search over all possible hyper-parameter configurations is prohibitively slow. Hyperband addresses this issue by eliminating a fraction of the configurations at every interval, allowing it to spend more time evaluating the most promising configurations. 

More specifically, given a total time budget $T$, Hyperband runs $n$ trials and eliminates a fraction $\eta$ of the remaining configurations per trial. Therefore, Hyperband spends constant time budget $T/n$ per trial and successively spends more time on the best configurations. There is a trade-off between testing a larger set of configurations for less time and testing a smaller set of configurations for more time. By varying the value of $n$, Hyperband is able to roughly estimate the optimal value of $n$, greatly improving its performance. The algorithm rests on the assumption that the target model is non-stochastic, but this is true for a variety of problem classes. 
