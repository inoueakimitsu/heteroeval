# heteroeval

<img src="https://github.com/inoueakimitsu/heteroeval/assets/2350154/7f4a8f92-2136-4442-aeb1-4737a5807f3d" width="50%">

[![Build Status](https://app.travis-ci.com/inoueakimitsu/heteroeval.svg?branch=main)](https://app.travis-ci.com/inoueakimitsu/heteroeval)
<a href="https://github.com/inoueakimitsu/heteroeval/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/inoueakimitsu/heteroeval"></a>

A Python package designed for the evaluation of machine learning models with heterogeneous test data.

## Features

Imagine a scenario where the observed data consists of multiple groups, and the composition of these groups changes in a non-stationary manner. If the expected value of a machine learning model's evaluation metric varies by group, and this expected value doesn't vary based on factors other than the group, the model's evaluation metric will fluctuate non-stationarily unless viewed group-by-group. This fluctuation complicates the comparison of evaluation metrics across different models. This library aids in automatically determining an appropriate grouping method for such scenarios, ensuring that if the model remains consistent, its evaluation metrics within each group will too.

### Detailed Usecase

Within the health application domain, it's crucial to monitor metrics like physical activity, dietary habits, and sleep patterns to forecast health risks. Given the diverse user base, ranging from teenagers to retirees in their 60s and from active athletes to office workers, prediction complexities can significantly differ between groups. Moreover, if a dataset doesn't have a balanced representation of each user group, certain groups might overly influence the results. This highlights the need to segment predictions and evaluations based on distinct user demographics.

However, overly detailed segmentation brings its own set of challenges. Segmenting users into numerous specific groups can lead to scarce evaluation data for each segment. Evaluating based on smaller datasets can result in greater metric variance, making it harder to accurately assess machine learning models.

To address this, it's essential to group users with the right level of granularity. heteroeval provides a solution by suggesting the best granularity for user grouping, considering evaluation metric trends and the amount of evaluation data, without depending on the actual feature values. For instance, if metrics for users in their 20s are similar to those in their 30s, heteroeval might advise clustering these age groups together.

By utilizing heteroeval, professionals can account for the unique evaluation metrics of different user groups, ensuring a more precise model evaluation.

## Mathematical Formulation

### 1. Evaluation Metric Calculation

For a given model $m$, regime $r$, group $G$, and data point $i$, we calculate the evaluation metric using a generic function $F$.

```math
E_{m,r,G} = F(y_{m,r,i}, \hat{y}_{m,r,i})
```

Where:
- $E_{m,r,G}$ represents the evaluation metric for model $m$, regime $r$, and group $G$.
- $y_{m,r,i}$ and $\hat{y}_{m,r,i}$ denote the true value and predicted value, respectively.
- $F$ is a general function to compute the evaluation metric. As an example, the squared error can be used and is represented as:

```math
F(y, \hat{y}) = \frac{1}{N_{r,G}} \sum_{i \in I_{r,G}} (y_{m,r,i} - \hat{y}_{m,r,i})^2
```

Here, $I_{r,G}$ is the index set for regime $r$ and group $G$.

### 2. Inter-group Evaluation Metric Variation

Given a grouping rule $g$, we compute the variation in evaluation metrics for each group $G$.

```math
V_{m, g, G} = \text{Aggregate}_{\text{inter-regime}}(E_{m,r_1,G}, E_{m,r_2,G}, \ldots, E_{m,r_{K},G})
```

Where:
- $\text{Aggregate}_{\text{inter-regime}}$ is a general function to aggregate evaluation metrics across regimes. An example implementation can be the standard deviation.

### 3. Cost Function

The cost function calculates the average of the evaluation metric variations $V_{m,g,G}$ for each group $G$ for a model $m$, and then aggregates these results across the model.

```math
C_{m,g} = \text{Aggregate}_{\text{group}}(V_{m,g,G_1}, V_{m,g,G_2}, \ldots, V_{m,g,G_{|G|}})
```

```math
C_m = \text{Aggregate}_{\text{model}}(C_{m,g_1}, C_{m,g_2}, \ldots, C_{m,g_{|G|}})
```

Where:
- $`\text{Aggregate}_{\text{group}}$ and $\text{Aggregate}_{\text{model}}`$ are general functions to aggregate evaluation metrics per group and across the model, respectively. An example implementation can be the average.

### 4. Optimization

We search for the grouping rule $g$ that minimizes the cost function $C_m$. Specifically, we generate grouping rules using different feature combinations, calculate the cost function for each rule, and select the rule that results in the lowest cost.

## Installation

```bash
pip install git+https://github.com/inoueakimitsu/heteroeval
```

## Usage

Simply call `find_best_grouping()`, as shown below:

```python
from heteroeval import find_best_grouping

find_best_grouping(
    n_models,
    regimes,
    X, y_true,
    y_pred_for_each_model,
    evaluation_measure,
    inter_regime_variation_measure,
    groupwise_variation_measure_aggregate_function,
    modelwise_variation_measure_aggregate_function,
    cost_function,
    optimizer)
```

Refer to `heteroeval/discrete.py` for a comprehensive working example.

## License

heteroeval is licensed under the MIT License.

