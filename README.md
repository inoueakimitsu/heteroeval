# heteroeval

[![Build Status](https://app.travis-ci.com/inoueakimitsu/heteroeval.svg?branch=main)](https://app.travis-ci.com/inoueakimitsu/heteroeval)
<a href="https://github.com/inoueakimitsu/heteroeval/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/inoueakimitsu/heteroeval"></a> 

Python package for evaluation of machine learning models in heterogeneous test data.

## Features

Consider a situation where the observed data is composed of multiple groups and the composition of the groups changes in a non-stationary pattern.
In this situation, when the expected value of an evaluation index of a machine learning model differs depending on the group, and the expected value of the evaluation index does not differ depending on factors other than the group, the evaluation index of this model will change non-stationarily unless we look at each group. This makes it impossible to compare the evaluation indices of multiple models.
This library automatically searches for an appropriate grouping method in the above situation, such that if the model does not change, the evaluation indices within the group will also not change.

## Installation

```bash
pip install git+https://github.com/inoueakimitsu/heteroeval
```

## Usage

only call find_best_grouping(), like this:

```python
from heteroeval import find_best_grouping

find_best_grouping(n_models, regimes, X, y_true, y_pred_for_each_model, evaluation_measure, inter_regime_variation_measure, groupwise_variation_measure_aggregate_function, modelwise_variation_measure_aggregate_function, cost_function, optimizer)

```

See `heteroeval/discrete.py` for an example of a fully working example.

## License

heteroeval is available under the MIT License.
