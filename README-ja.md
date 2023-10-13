# heteroeval

<img src="https://github.com/inoueakimitsu/heteroeval/assets/2350154/7f4a8f92-2136-4442-aeb1-4737a5807f3d" width="50%">

[![Build Status](https://app.travis-ci.com/inoueakimitsu/heteroeval.svg?branch=main)](https://app.travis-ci.com/inoueakimitsu/heteroeval)
<a href="https://github.com/inoueakimitsu/heteroeval/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/inoueakimitsu/heteroeval"></a>

heteroeval は、異質なテストデータを持つ機械学習モデルの評価のために設計された Python パッケージです。

## 機能

複数のグループで構成される観測データがあり、これらのグループの構成が非定常的に変化するシナリオを想像してください。機械学習モデルの評価指標の期待値がグループによって異なり、この期待値がグループ以外の要因に基づかない場合、モデルの評価指標はグループごとに見ない限り非定常的に変動します。この変動は、異なるモデル間での評価指標の比較を複雑にします。このライブラリは、モデルが一貫していれば、各グループ内の評価指標も一貫していることを保証するため、このようなシナリオのための適切なグループ化方法を自動的に決定するのに役立ちます。

### 詳細な使用例

健康アプリケーションの分野では、健康リスクを予測するために、身体活動、食習慣、睡眠パターンなどの指標を監視することが重要です。10代の若者から60代の退職者、アクティブなアスリートからオフィスワーカーまで、ユーザー基盤は多様です。そのため、グループ間での予測の複雑さが大きく異なることがあります。さらに、データセットが各ユーザーグループを均等に表現していない場合、特定のグループが結果に過度な影響を及ぼす可能性があります。これは、ユーザーの人口統計に基づいて予測と評価をセグメント化する必要性を強調しています。

しかし、詳細すぎるセグメンテーションは、独自の課題を持っています。ユーザーを多数の特定のグループにセグメント化することは、各セグメントの評価データが少なくなる可能性があります。小さいデータセットを基に評価することは、指標の変動が大きくなり、機械学習モデルを正確に評価することが難しくなる可能性があります。

この問題に対処するためには、適切な粒度でユーザーをグループ化することが不可欠です。heteroeval は、評価指標のトレンドや評価データの量を考慮して、実際の特徴値に依存せずにユーザーのグループ化のための最適な粒度を提案するソリューションを提供します。例えば、20代のユーザーの指標が30代のユーザーのものと似ている場合、heteroeval は、これらの年齢グループをまとめることを提案するかもしれません。

heteroeval を使用することで、専門家は異なるユーザーグループのユニークな評価指標を考慮に入れ、より正確なモデルの評価を確保することができます。

## 数学的な定式化

### 1. 評価指標の計算

与えられたモデル $m$、レジーム $r$、グループ $G$、データポイント $i$ に対して、一般的な関数 $F$ を使用して評価指標を計算します。

```math
E_{m,r,G} = F(y_{m,r,i}, \hat{y}_{m,r,i})
```

ここで:
- $E_{m,r,G}$ は、モデル $m$、レジーム $r$、グループ $G$ の評価指標を表します。
- $y_{m,r,i}$ と $\hat{y}_{m,r,i}$ は、それぞれ真の値と予測値を示します。
- $F$ は、評価指標を計算する一般的な関数です。例として、二乗誤差を使用することができ、以下のように表されます:

```math
F(y, \hat{y}) = \frac{

1}{N_{r,G}} \sum_{i \in I_{r,G}} (y_{m,r,i} - \hat{y}_{m,r,i})^2
```

ここで、$I_{r,G}$ は、レジーム $r$ およびグループ $G$ のインデックスセットです。

### 2. グループ間の評価指標の変動

グルーピングルール $g$ が与えられた場合、各グループ $G$ の評価指標の変動を計算します。

```math
V_{m, g, G} = \text{Aggregate}_{\text{inter-regime}}(E_{m,r_1,G}, E_{m,r_2,G}, \ldots, E_{m,r_{K},G})
```

ここで:
- $\text{Aggregate}_{\text{inter-regime}}$ は、レジーム間での評価指標を集約する一般的な関数です。一つの実装例として、標準偏差が考えられます。

### 3. コスト関数

コスト関数は、モデル $m$ の各グループ $G$ の評価指標の変動 $V_{m,g,G}$ の平均を計算し、これらの結果をモデル全体で集約します。

```math
C_{m,g} = \text{Aggregate}_{\text{group}}(V_{m,g,G_1}, V_{m,g,G_2}, \ldots, V_{m,g,G_{|G|}})
```

```math
C_m = \text{Aggregate}_{\text{model}}(C_{m,g_1}, C_{m,g_2}, \ldots, C_{m,g_{|G|}})
```

ここで:
- $\text{Aggregate}_{\text{group}}$ および $\text{Aggregate}_{\text{model}}$ は、グループごとおよびモデル全体での評価指標を集約する一般的な関数です。一つの実装例として、平均が考えられます。

### 4. グルーピング

グルーピングのプロセスは、特徴量とメタ情報によって特徴付けられるデータサンプルを、特定のグループインデックスに変換することを含みます。このマッピングは、$\theta$ によってパラメータ化された関数によって表されます:

```math
G_i = g_{\theta}(x_i, m_i)
```

ここで:
- $G_i$ は、$i$ 番目のサンプルのグループインデックスを表します。
- $x_i$ は、$i$ 番目のサンプルの特徴ベクトルです。
- $m_i$ は、$i$ 番目のサンプルに関連するメタ情報を示します。
- $g_{\theta}$ は、特徴とメタ情報に基づいてグループインデックスを決定するグルーピング関数で、$\theta$ によってパラメータ化されます。

### 5. 最適化

我々の目的は、コスト関数 $C$ を最小化するパラメータ $\theta$ を見つけることです。具体的には、$\theta$ を変更することで、異なるグルーピングルールを生成し、それぞれのコスト関数を計算します。最小のコストをもたらす $\theta$ を選択します:

```math
\theta^* = \arg\min_{\theta} C = \arg\min_{\theta} \text{Aggregate}_{\text{grouping}}(C_{m1}, C_{m2}, \ldots)
```

$\theta^*$ を見つけることで、最適なグルーピングルールを決定します。

## インストール

```bash
pip install git+https://github.com/inoueakimitsu/heteroeval
```

## 使用方法

以下に示すように、`find_best_grouping()` を呼び出してください:

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

詳細な実際の例については、`heteroeval/discrete.py` を参照してください。

## ライセンス

heteroeval は、MIT ライセンスの下でライセンスされています。
