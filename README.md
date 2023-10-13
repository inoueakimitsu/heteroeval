# heteroeval
[![Build Status](https://app.travis-ci.com/inoueakimitsu/heteroeval.svg?branch=main)](https://app.travis-ci.com/inoueakimitsu/heteroeval)
<a href="https://github.com/inoueakimitsu/heteroeval/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/inoueakimitsu/heteroeval"></a> 


<img src="https://github.com/inoueakimitsu/heteroeval/assets/2350154/07ebbf9b-7c1a-400f-8fbd-91af0449e867" width="30%">

Python package for evaluation of machine learning models in heterogeneous test data.

## Features

Consider a situation where the observed data is composed of multiple groups and the composition of the groups changes in a non-stationary pattern.
In this situation, when the expected value of an evaluation index of a machine learning model differs depending on the group, and the expected value of the evaluation index does not differ depending on factors other than the group, the evaluation index of this model will change non-stationarily unless we look at each group. This makes it impossible to compare the evaluation indices of multiple models.
This library automatically searches for an appropriate grouping method in the above situation, such that if the model does not change, the evaluation indices within the group will also not change.

## Detailed Usecase

In the realm of health applications, tracking metrics such as physical activity, dietary habits, and sleep patterns is essential to predict health risks. Considering the spectrum of users – from students in their teens to retirees in their 60s, from active athletes to desk-bound workers – the complexity of predictions can vary dramatically between groups. Furthermore, if the proportion of each user group is not balanced within a dataset, the influence of particular groups may become predominant. This underscores the imperative of segmenting predictions and evaluations according to distinct user groups.

However, a granular segmentation poses its own challenges. Dividing users into too many fine-grained groups can result in limited evaluation data for each group. When evaluating on a smaller dataset, the variance in evaluation metrics increases, complicating the accurate assessment of machine learning models.

Addressing this challenge requires grouping users at an appropriate granularity. **HeteroEval** offers a solution by automatically suggesting the optimal granularity for user grouping, based on evaluation metric trends and the volume of evaluation data, without relying on the feature values themselves. For instance, if the evaluation metrics for users in their 20s closely resemble those in their 30s, **HeteroEval** might recommend treating these age groups as a single cluster.

By leveraging **HeteroEval**, practitioners can consider the distinct evaluation metrics across user groups, facilitating a more accurate model assessment.

## Installation

```bash
pip install git+https://github.com/inoueakimitsu/heteroeval
```

## Usage

only call find_best_grouping(), like this:

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

See `heteroeval/discrete.py` for an example of a fully working example.

## License

heteroeval is available under the MIT License.
