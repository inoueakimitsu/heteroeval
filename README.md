# heteroeval

<img src="https://github.com/inoueakimitsu/heteroeval/assets/2350154/07ebbf9b-7c1a-400f-8fbd-91af0449e867" width="30%">

[![Build Status](https://app.travis-ci.com/inoueakimitsu/heteroeval.svg?branch=main)](https://app.travis-ci.com/inoueakimitsu/heteroeval)
<a href="https://github.com/inoueakimitsu/heteroeval/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/inoueakimitsu/heteroeval"></a>

A Python package designed for the evaluation of machine learning models with heterogeneous test data.

**Features**

Imagine a scenario where the observed data consists of multiple groups, and the composition of these groups changes in a non-stationary manner. If the expected value of a machine learning model's evaluation metric varies by group, and this expected value doesn't vary based on factors other than the group, the model's evaluation metric will fluctuate non-stationarily unless viewed group-by-group. This fluctuation complicates the comparison of evaluation metrics across different models. This library aids in automatically determining an appropriate grouping method for such scenarios, ensuring that if the model remains consistent, its evaluation metrics within each group will too.

**Detailed Usecase**

Within the health application domain, it's crucial to monitor metrics like physical activity, dietary habits, and sleep patterns to forecast health risks. Given the diverse user base, ranging from teenagers to retirees in their 60s and from active athletes to office workers, prediction complexities can significantly differ between groups. Moreover, if a dataset doesn't have a balanced representation of each user group, certain groups might overly influence the results. This highlights the need to segment predictions and evaluations based on distinct user demographics.

However, overly detailed segmentation brings its own set of challenges. Segmenting users into numerous specific groups can lead to scarce evaluation data for each segment. Evaluating based on smaller datasets can result in greater metric variance, making it harder to accurately assess machine learning models.

To address this, it's essential to group users with the right level of granularity. HeteroEval provides a solution by suggesting the best granularity for user grouping, considering evaluation metric trends and the amount of evaluation data, without depending on the actual feature values. For instance, if metrics for users in their 20s are similar to those in their 30s, HeteroEval might advise clustering these age groups together.

By utilizing HeteroEval, professionals can account for the unique evaluation metrics of different user groups, ensuring a more precise model evaluation.

**Installation**

```bash
pip install git+https://github.com/inoueakimitsu/heteroeval
```

**Usage**

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

**License**

heteroeval is licensed under the MIT License.

