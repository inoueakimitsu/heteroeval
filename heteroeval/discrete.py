import numpy as np
import pandas as pd
import scipy.optimize


def generate_random_binary_features(n):
    return np.random.choice([0, 1], size=(n, ), p=[1/2, 1/2])


n_latent_groups = 10
n_features_depends_on_groups = 7
n_features_independent_of_groups = 20
n_change_group_proportion = 5
alpha_for_group_proportion_dirchlet_distribution = np.ones(n_latent_groups)
n_sample_of_each_group_proportion_changes = 100
n_models = 5

features_depends_on_groups = []
for i_latent_group in range(n_latent_groups):
    features_depends_on_groups.append(
        generate_random_binary_features(n_features_depends_on_groups))

rows = []
latent_groups = []
regimes = []
for i_regime_for_change_group_proportion in range(n_change_group_proportion):
    group_proportion = np.random.dirichlet(
        alpha=alpha_for_group_proportion_dirchlet_distribution)
    groups_of_current_regime = np.random.choice(
        a=np.arange(n_latent_groups), size=n_sample_of_each_group_proportion_changes, p=group_proportion)
    for group in groups_of_current_regime:
        features_independent_of_groups = generate_random_binary_features(
            n_features_independent_of_groups)
        features_all = np.concatenate(
            [features_depends_on_groups[group], features_independent_of_groups], axis=0)
        rows.append(features_all)
        latent_groups.append(group)
    regimes.extend(np.ones(n_sample_of_each_group_proportion_changes)
                   * i_regime_for_change_group_proportion)
latent_groups = np.array(latent_groups)
regimes = np.array(regimes)
X = np.stack(rows)

centroids_for_groups = np.random.normal(size=n_latent_groups, scale=5)
y_true = np.random.normal(loc=centroids_for_groups[latent_groups])

y_pred_for_each_model = []
for i_model in range(n_models):
    groupwise_bias_of_y_in_current_model = np.random.normal(
        size=n_latent_groups, scale=0.5)
    y_pred_for_each_model.append(
        y_true + groupwise_bias_of_y_in_current_model[latent_groups] + np.random.normal(scale=0.1, size=y_true.shape))


def evaluation_measure(predict, target) -> float:
    return np.mean((predict - target)**2)


def inter_regime_variation_measure(evaluation_measures_in_regime) -> float:
    return np.std(evaluation_measures_in_regime) if len(evaluation_measures_in_regime) > 3 else np.inf


def groupwise_variation_measure_aggregate_function(groupwise_variation_measures) -> float:
    return np.mean(groupwise_variation_measures)


def modelwise_variation_measure_aggregate_function(modelwise_variation_measures) -> float:
    return np.mean(modelwise_variation_measures)


predicted_groups = latent_groups + X[:, 1]


def cost_function(n_models, regimes, y_true, y_pred_for_each_model, evaluation_measure, inter_regime_variation_measure, groupwise_variation_measure_aggregate_function, modelwise_variation_measure_aggregate_function, predicted_groups):
    modelwise_variation_measures = []
    for i_model in range(n_models):
        df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred_for_each_model[i_model],
            "regime": regimes,
            "group_predicted": predicted_groups,
        })
        df_grouped_predicted = df.groupby(["group_predicted", "regime"]).apply(
            lambda x: evaluation_measure(x["y_pred"].values, x["y_true"].values))
        df_groupwise_variation = df_grouped_predicted.groupby(
            "group_predicted").apply(inter_regime_variation_measure)
        modelwise_variation_measures.append(
            groupwise_variation_measure_aggregate_function(df_groupwise_variation.values))
    return modelwise_variation_measure_aggregate_function(modelwise_variation_measures)


def optimizer(X, _cost_function):
    n_iter = 100
    best_group = None
    best_cost = np.inf
    for _ in range(n_iter):
        candidate = np.random.choice([0, 1], size=X.shape[1], p=[0.8, 0.2])
        print(candidate)
        X_sub = X[:, candidate == 1]
        n_col_X_sub = X_sub.shape[1]
        groups_X = np.sum(X_sub * (2**(np.arange(n_col_X_sub)+1)), axis=1)
        current_cost = _cost_function(groups_X)
        print(current_cost)
        if current_cost < best_cost:
            best_group = candidate
            best_cost = current_cost
    return best_group, best_cost


def find_best_grouping(n_models, regimes, X, y_true, y_pred_for_each_model, evaluation_measure, inter_regime_variation_measure, groupwise_variation_measure_aggregate_function, modelwise_variation_measure_aggregate_function, cost_function, optimizer):
    def _cost_function(group_candidate): return cost_function(n_models, regimes, y_true, y_pred_for_each_model, evaluation_measure, inter_regime_variation_measure,
                                                              groupwise_variation_measure_aggregate_function, modelwise_variation_measure_aggregate_function, predicted_groups=group_candidate)
    return optimizer(X, _cost_function)


if __name__ == "__main__":
    print(find_best_grouping(n_models, regimes, X, y_true, y_pred_for_each_model, evaluation_measure, inter_regime_variation_measure,
                       groupwise_variation_measure_aggregate_function, modelwise_variation_measure_aggregate_function, cost_function, optimizer))
