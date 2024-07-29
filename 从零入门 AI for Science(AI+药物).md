# Datawhale AI 夏令营-siRNA药物药效预测

学习者手册:[https://exn8g66dnwu.feishu.cn/docx/T7WGd7goqowRvFxwoApclo9Pn0b](https://linklearner.com/activity/12/4/4)

赛事地址(数据集下载):https://challenge.xfyun.cn/topic/info?type=electricity-demand&option=ssgy&ch=dw24_uGS8Gs

你可以在[power_predict_ipynb](https://github.com/machenme/Datawhale/blob/main/power_predict/main.ipynb)看到我的代码
<details>

<summary>task1:初步了解项目</summary>

## 数据类型

赛题数据由训练集和测试集组成, 为了保证比赛的公平性, 将每日日期进行脱敏, 用1-N进行标识, 即1为数据集最近一天, 其中1-10为测试集数据.

|特征字段|	字段描述|
|--|--|
|id	|房屋id|
|dt	|日标识|
|type	|房屋类型|
|target	|实际电力消耗, 预测目标|

## 完整代码:

```python
# 1. 导入需要用到的相关库
# 导入 pandas 库, 用于数据处理和分析
import pandas as pd
# 导入 numpy 库, 用于科学计算和多维数组操作
import numpy as np

# 2. 读取训练集和测试集
# 使用 read_csv() 函数从文件中读取训练集数据, 文件名为 'train.csv'
train = pd.read_csv('./data/data283931/train.csv')
# 使用 read_csv() 函数从文件中读取测试集数据, 文件名为 'train.csv'
test = pd.read_csv('./data/data283931/test.csv')

# 3. 计算训练数据最近11-20单位时间内对应id的目标均值
target_mean = train[train['dt']<=20].groupby(['id'])['target'].mean().reset_index()

# 4. 将target_mean作为测试集结果进行合并
test = test.merge(target_mean,  on=['id'],  how='left')

# 5. 保存结果文件到本地
test[['id', 'dt', 'target']].to_csv('submit.csv',  index=None)
```

## 学习代码

```python
target_mean = train[train['dt']<=20].groupby(['id'])['target'].mean().reset_index()
```
- `train[train['dt']<=20]`读取`train`中`dt`列小于等于20的所有数据

- `groupby`函数将数据进行分组然后再进行下一步操作
    - 通过`groupby(['id'])`告诉系统以`id`这列数据进行分组, `id`相同的数据均会被分到一个组里.
    - `groupby(['id'])['target']`则是分组之后只需要`target`这列数据
    - `groupby(['id'])['target'].mean()`获取每个分组的`target`的平均值
- `reset_index()`重建数据索引

```python
test = test.merge(target_mean,  on=['id'],  how='left')
```
`merge`函数用来合并两个DateFrame
- `df1.merge(df2)`与`pd.merge(df1, df2)`是等价的, 都是合并`df1`与`df2`数据
- `on`以哪一列作为合并的依据, 这里以`id`列作为合并的依据
- `how`如何合并, `left`保留左侧df1的所有行, 如果右侧df2中没有匹配的键, 则相应的列将填充为 NaN
    - `right`以右侧为准, 如果左侧没有对应的数据填充NaN
    - `inner`取交集, 合并后的数据只有左右两个df都有的部分
    - `outer`取并集, 保留二者所有行, 没有的部分填充NaN

显然我们不能够简单的用过去11天到20天的平均值作为过去1到10天的预测依据.我们应该找到更好的预测手段.

</details>

<details>

<summary>Task2:特征工程入门</summary>

随着昨天运行了baseline之后, 出现了新的问题, 对于本次数据是否存在一些规律性, 例如按照某些时间间隔出现周期性重复?比如常见的7天, 30天, 90天等等.数据只有500个日期左右, 大于90天的周期性也许意义不大了.

或者某些特征会对最终的预测有着更大的影响, 而某些特征可能对实际预测结果基本没有意义甚至是负面影响?
```python
# 先查看数据是否存在规律
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

def draw_pic(data):
    unique_ids = data["id"].unique()
    plt.figure(figsize=(12, 6))

    for id in unique_ids[:5]:
        demo_data = data[data["id"] == id]
        # 为了更直观理解图像, 将target翻转, 题目含义是距离当天的时间, 那么反过来就是历史时间, 
        # 例如0就会对应第一天, 1对应第二天, 一直到496天, 然后通过496天数据来预测未来10天数据.
        plt.plot(demo_data["target"].tolist()[::-1],  label=f"ID: {id}")
    plt.legend()
    plt.title('不同ID对应的每日用电量')
    plt.xlabel('日期')
    plt.ylabel('每日用电量')
    plt.show()

draw_pic(train)
```
![任意5个用户](https://github.com/machenme/Datawhale/blob/main/power_predict/imgs/output.png)

```python
# 也许不同的type类型也有影响?
same_type_train = train[train['type']==0]
draw_pic(same_type_train)
```
![相同Type下任意5个用户](https://github.com/machenme/Datawhale/blob/main/power_predict/imgs/output_same_id.png)



查看不同Type对应的平均用电量
![不同Type对应的平均用电量](https://github.com/machenme/Datawhale/blob/main/power_predict/imgs/diff_type_power_usage.png)



## LightGBM
- LightGBM 是一个梯度 boosting 框架,  使用基于学习算法的决策树. 它是分布式的,  高效的,  装逼的,  它具有以下优势: **速度和内存使用的优化** **减少分割增益的计算量** **通过直方图的相减来进行进一步的加速** **减少内存的使用** **减少并行学习的通信代价** ... 反正就是很多优点
- 项目地址: https://lightgbm.cn/

## 如何使用LightGBM
### 导入对应的软件包
```python
    # 直接上工具LightGBM
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    mean_squared_log_error, 
    mean_absolute_error, 
    mean_squared_error, 
)
import tqdm
import sys
import os
import gc
import argparse
import warnings

warnings.filterwarnings("ignore")
```
- 导入对应的数据集
```python
train = pd.read_csv("./dataset/train.csv")
test = pd.read_csv("./dataset/test.csv")
```
### 特征工程
```python
# 合并训练数据和测试数据, 并进行排序
data = pd.concat([test,  train],  axis=0,  ignore_index=True)
data = data.sort_values(['id', 'dt'],  ascending=False).reset_index(drop=True)

# 历史平移
for i in range(10, 30):
    data[f'last{i}_target'] = data.groupby(['id'])['target'].shift(i)
    
# 窗口统计
data[f'win3_mean_target'] = (data['last10_target'] + data['last11_target'] + data['last12_target']) / 3

# 进行数据切分
train = data[data.target.notnull()].reset_index(drop=True)
test = data[data.target.isnull()].reset_index(drop=True)

# 确定输入特征
train_cols = [f for f in data.columns if f not in ['id', 'target']]
```
- `pd.concat([test,  train],  axis=0,  ignore_index=True)`
    - `pd.concat` 堆叠数据, 与`merge`不同, 不需要任何依据
    - `axis=0,  ignore_index=True` 按照行合并.`axis=1`则是按照列合并.`ignore_index=True`则是忽略原来的索引
- `sort_values(['id', 'dt'],  ascending=False).reset_index(drop=True)`
    - `sort_values` 对`DateFrame`数据进行排序, 后面跟排序依据
    - `['id', 'dt'],  ascending=False` 以 `id`作为第一排序依据, 如果`id`相同, 再用`dt`作为第二排序依据, 并且`ascending=False`说明不需要反转, 那么默认就是从大到小排序, 也就是降序排列.
    - `reset_index(drop=True)` 排序后重建`DateFrame`索引并且舍弃原来的索引.

```python
for i in range(10, 30):
    data[f'last{i}_target'] = data.groupby(['id'])['target'].shift(i)
```
- `shift(i)`将数据下移`i`行, 前面不足的部分用`NaN`填充后新建一列存在数据的最后

|id|	dt	|type	|target	|last10_target	|last11_target|	last12_target|	last13_target	|last14_target|	last15_target|	
|--|--|--|--|--|--|--|--|--|--|
|fff81139a7|	496|	5|	23.288|	18.145|	NaN|	NaN|	NaN|	NaN	|NaN|
|fff81139a7	|495|	5|	25.252|	22.021	|18.145|	NaN|	NaN	|NaN|	NaN
|fff81139a7	|494|	5|	16.963|	21.282|	22.021|	18.145	|NaN|	NaN|	NaN|
|ff81139a7|	493|	5|	29.759|	22.818|	21.282|	22.021|	18.145|NaN|NaN|

```python
data[f'win3_mean_target'] = (data['last10_target'] + data['last11_target'] + data['last12_target']) / 3
```
- 将过去三天(今天的前第10天, 第11天, 第12天)的数据求平均值

```python
# 进行数据切分
train = data[data.target.notnull()].reset_index(drop=True)
test = data[data.target.isnull()].reset_index(drop=True)
```
- `notnull()` 只保留非空行的数据

注意的训练集和验证集的构建:因为数据存在时序关系, 所以需要严格按照时序进行切分, 并且时间序列问题只能是过去的事情对未来造成影响, 反过来则没有意义

### 创建模型
```python
def time_model(lgb,  train_df,  test_df,  cols):
    # 训练集和验证集切分
    trn_x,  trn_y = train_df[train_df.dt>=31][cols],  train_df[train_df.dt>=31]['target']
    val_x,  val_y = train_df[train_df.dt<=30][cols],  train_df[train_df.dt<=30]['target']
    # 构建模型输入数据
    train_matrix = lgb.Dataset(trn_x,  label=trn_y)
    valid_matrix = lgb.Dataset(val_x,  label=val_y)
    # lightgbm参数
    lgb_params = {
        'boosting_type': 'gbdt', 
        'objective': 'regression', 
        'metric': 'mse', 
        'min_child_weight': 5, 
        'num_leaves': 2 ** 5, 
        'lambda_l2': 10, 
        'feature_fraction': 0.8, 
        'bagging_fraction': 0.8, 
        'bagging_freq': 4, 
        'learning_rate': 0.05, 
        'seed': 2024, 
        'nthread' : 16, 
        'verbose' : -1, 
    }
    # 训练模型
    model = lgb.train(lgb_params,  train_matrix,  50000,  valid_sets=[train_matrix,  valid_matrix],  
                      categorical_feature=[],  verbose_eval=500,  early_stopping_rounds=500)
    # 验证集和测试集结果预测
    val_pred = model.predict(val_x,  num_iteration=model.best_iteration)
    test_pred = model.predict(test_df[cols],  num_iteration=model.best_iteration)
    # 离线分数评估
    score = mean_squared_error(val_pred,  val_y)
    print(score)
       
    return val_pred,  test_pred
    
lgb_oof,  lgb_test = time_model(lgb,  train,  test,  train_cols)

# 保存结果文件到本地
test['target'] = lgb_test
test[['id', 'dt', 'target']].to_csv('submit.csv',  index=None)
```
- LightGBM 可以直接使用 categorical features(分类特征)作为 input(输入). 它不需要被转换成 `one-hot coding(独热编码)`,  并且它比独热编码更快(约快上 8 倍)
    - 在构造 `Dataset` 之前,  应该将分类特征转换为 `int` 类型的值.

自己添加了L1正则化优化, 结果发现不能乱改, 改了反而降低了准确率QaQ.后面希望有更准确的做法

更新:当回溯周期提高到90天后, 获得了稍好的成绩, 哈哈哈哈



</details>
<details>

<summary>Task3:特征优化与深度学习模型</summary>

Task2中了解了Lightgbm与前10天数据以及3天融合数据一起工作并不能取得较好的结果.也许需要多种模型融合或者直接使用神经网络LSTM之类的时间序列模型.

## 什么是时间特征?
- 在深度学习中, 时间特征通常指的是数据集中与时间相关的属性或模式, 它们可以用于预测或分类任务.时间特征可以是连续的, 也可以是离散的, 它们可以反映数据随时间变化的趋势、周期性或季节性.
    - 通过`特征工程`挖掘特征可以很快的提升模型预测效果
![时间特征](https://github.com/machenme/Datawhale/blob/main/power_predict/imgs/time-series.png)

## 特征优化
![项目流程](https://github.com/machenme/Datawhale/blob/main/power_predict/imgs/liucheng.png)

- 模型优化: 不同模型, 不同超参数, 模型相互组合
- 特征优化: 尝试提取更多特征
    - 历史平移: 通过将时间序列数据中的每个时间点的值向前或向后移动一定数量的周期来创建的.例如, 如果我们有一个时间序列 $t_1,  t_2,  ...,  t_n $ , 我们可以创建平移特征 $t_{t-1},  t_{t-2},  ...,  t_{t-k}$, 其中 $k$ 是平移的周期数.这些特征可以帮助模型理解时间序列中过去的值如何影响当前值, 从而捕捉时间依赖性.

    - 差分特征: 通过计算连续时间点之间的差异来创建的.对于时间序列$t_1,  t_2,  ...,  t_n$, 一阶差分可以表示为$\Delta t_t = t_t - t_{t-1}$.
        - 差分可以减少时间序列的非平稳性, 即消除或减少序列的均值和方差随时间变化的特性.这有助于突出时间序列的趋势或季节性变化.
        - 高阶差分可以通过连续计算差分来实现, 例如, 二阶差分是一阶差分的差分.
    - 窗口统计特征: 在时间序列分析中, 对数据的特定时间窗口（连续的时间段）进行统计分析, 以提取该窗口内数据的某些统计属性.这些特征可以捕捉时间序列的局部特性, 趋势、波动性、周期性等.

合并训练数据和测试数据,对 `id` `dt` 依次排序 `id`优先
```python

data = pd.concat([train, test], axis=0).reset_index(drop=True)
data = data.sort_values(['id','dt'], ascending=False).reset_index(drop=True)
```
以`id`为分组依据之后,找到每组的`target`数据,然后`shift`从第10天到35天一共26列数据
```python
for i in range(10,36):
    # 以id为分组依据之后,找到每组的
    data[f'target_shift{i}'] = data.groupby('id')['target'].shift(i)
```
对`shift10`进行1到3天的差分数据提取更多有用信息
```python

for i in range(1,4):
    data[f'target_shift10_diff{i}'] = data.groupby('id')['target_shift10'].diff(i)
```

`rolling`函数用于创建一个滚动窗口

- `window` 滚动窗口大小
- `min_periods` 定义了进行计算所需的最小元素数量. 如果窗口中的元素数量少于`min_periods`, 那么结果将被标记为NaN. 这个参数可以防止在窗口开始或结束时, 由于数据不足而产生误导性的结果.
- `closed` 决定了包含数据哪个端点, 默认是`left`左闭右开. 还有`right` `both` `none`

与`Lightgbm`类似,不过这次采用了三种模型混合,分别是`Lightgbm` `xgboost` `catboost`

- LightGBM 是一个基于梯度提升框架的高效、分布式、高性能的机器学习算法，它使用基于树的学习算法，特别适合处理大规模数据集。
    - 在时间序列预测中，LightGBM 可以处理高维数据，并且通常能够快速收敛，提供准确的预测结果。

- XGBoost（eXtreme Gradient Boosting）是一种优化的梯度提升库，它设计用于提高树算法的性能，特别是在计算速度和准确性方面。
    - 在时间序列分析中，XGBoost 能够处理缺失值，并且支持自定义的树模型结构，使其在预测任务中表现出色。

- CatBoost 是一种先进的梯度提升算法，它特别擅长处理分类特征（categorical features），并提供了对类别特征的内置支持。
    - 在时间序列预测中，CatBoost 能够自动处理时间序列中的类别特征，并提供稳定和准确的预测。

### 三种模型优劣势
- LightGBM 的优势在于其速度和低内存使用，适合处理大规模数据集，但可能需要更多的调参来优化模型。
- XGBoost 的优势在于其灵活性和对缺失值的良好处理，但可能在某些情况下比LightGBM慢。
- CatBoost 的优势在于对类别特征的自动处理和稳定性，但可能在处理非类别特征时不如LightGBM和XGBoost高效。

事实上三种模型拟合之后结果确实有一定的提升,但是幅度不大,从252提升到了235左右,看来得另辟蹊径
<details>

<summary>模型代码</summary>

```python
# 窗口统计
for win in [15,30,50,70]:
    data[f'target_win{win}_mean'] = data.groupby('id')['target'].rolling(window=win, min_periods=3, closed='left').mean().values
    data[f'target_win{win}_max'] = data.groupby('id')['target'].rolling(window=win, min_periods=3, closed='left').max().values
    data[f'target_win{win}_min'] = data.groupby('id')['target'].rolling(window=win, min_periods=3, closed='left').min().values
    data[f'target_win{win}_std'] = data.groupby('id')['target'].rolling(window=win, min_periods=3, closed='left').std().values
# 历史平移 + 窗口统计
for win in [7,14,28,35,50,70]:
    data[f'target_shift10_win{win}_mean'] = data.groupby('id')['target_shift10'].rolling(window=win, min_periods=3, closed='left').mean().values
    data[f'target_shift10_win{win}_max'] = data.groupby('id')['target_shift10'].rolling(window=win, min_periods=3, closed='left').max().values
    data[f'target_shift10_win{win}_min'] = data.groupby('id')['target_shift10'].rolling(window=win, min_periods=3, closed='left').min().values
    data[f'target_shift10_win{win}_sum'] = data.groupby('id')['target_shift10'].rolling(window=win, min_periods=3, closed='left').sum().values
    data[f'target_shift710win{win}_std'] = data.groupby('id')['target_shift10'].rolling(window=win, min_periods=3, closed='left').std().values
```
```python
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def cv_model(clf, train_x, train_y, test_x, clf_name, seed=2024):
    """
    clf：调用模型
    train_x：训练数据
    train_y：训练数据对应标签
    test_x：测试数据
    clf_name：选择使用模型名
    seed：随机种子
    """
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(train_x.shape[0])
    test_predict = np.zeros(test_x.shape[0])
    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print(
            "************************************ {} ************************************".format(
                str(i + 1)
            )
        )
        trn_x, trn_y, val_x, val_y = (
            train_x.iloc[train_index],
            train_y[train_index],
            train_x.iloc[valid_index],
            train_y[valid_index],
        )

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)
            params = {
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": "mae",
                "min_child_weight": 6,
                "num_leaves": 2**6,
                "lambda_l2": 10,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 4,
                "learning_rate": 0.1,
                "seed": 2023,
                "nthread": 16,
                "verbose": -1,
            }
            model = clf.train(
                params,
                train_matrix,
                1000,
                valid_sets=[train_matrix, valid_matrix],
                categorical_feature=[],
                verbose_eval=200,
                early_stopping_rounds=100,
            )
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)

        if clf_name == "xgb":
            xgb_params = {
                "booster": "gbtree",
                "objective": "reg:squarederror",
                "eval_metric": "mae",
                "max_depth": 5,
                "lambda": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.7,
                "eta": 0.1,
                "tree_method": "hist",
                "seed": 520,
                "nthread": 16,
            }
            train_matrix = clf.DMatrix(trn_x, label=trn_y)
            valid_matrix = clf.DMatrix(val_x, label=val_y)
            test_matrix = clf.DMatrix(test_x)

            watchlist = [(train_matrix, "train"), (valid_matrix, "eval")]

            model = clf.train(
                xgb_params,
                train_matrix,
                num_boost_round=1000,
                evals=watchlist,
                verbose_eval=200,
                early_stopping_rounds=100,
            )
            val_pred = model.predict(valid_matrix)
            test_pred = model.predict(test_matrix)

        if clf_name == "cat":
            params = {
                "learning_rate": 0.1,
                "depth": 5,
                "bootstrap_type": "Bernoulli",
                "random_seed": 2023,
                "od_type": "Iter",
                "od_wait": 100,
                "random_seed": 11,
                "allow_writing_files": False,
            }

            model = clf(iterations=1000, **params)
            model.fit(
                trn_x,
                trn_y,
                eval_set=(val_x, val_y),
                metric_period=200,
                use_best_model=True,
                cat_features=[],
                verbose=1,
            )

            val_pred = model.predict(val_x)
            test_pred = model.predict(test_x)

        oof[valid_index] = val_pred
        test_predict += test_pred / kf.n_splits

        score = mean_absolute_error(val_y, val_pred)
        cv_scores.append(score)
        print(cv_scores)

    return oof, test_predict


# 选择lightgbm模型
lgb_oof, lgb_test = cv_model(
    lgb, train[train_cols], train["target"], test[train_cols], "lgb"
)
# 选择xgboost模型
xgb_oof, xgb_test = cv_model(
    xgb, train[train_cols], train["target"], test[train_cols], "xgb"
)
# 选择catboost模型
cat_oof, cat_test = cv_model(
    CatBoostRegressor, train[train_cols], train["target"], test[train_cols], "cat"
)

# 进行取平均融合
final_test = (lgb_test + xgb_test + cat_test) / 3
```
</details>

## Stacking融合与加权平均
Stacking融合和简单的加权平均是两种不同的模型融合技术二者都在机器学习中用于提高预测的准确性
- Stacking 是一种更复杂的模型集成技术，它使用多个不同的模型作为基模型，并将它们的预测结果作为输入来训练一个新的模型，称为元模型（meta-model）或顶层模型。
    - 在Stacking中，基模型可以是不同类型的算法，例如决策树、神经网络、支持向量机等，而不仅仅是梯度提升机。
    - Stacking通常涉及两个或多个层次的模型：第一层是基模型，第二层是元模型，后者学习如何最好地组合基模型的预测。


- 加权平均是一种更简单的模型融合方法，它直接将不同模型的预测结果按照一定的权重进行平均。
    - 这种方法不需要训练额外的模型，只需要确定每个模型预测结果的权重，然后计算加权和。
    - 加权平均通常不考虑模型之间的复杂关系，而是简单地将它们视为独立的预测器。

- 异同点
    - 模型复杂性：Stacking融合通常比加权平均更复杂，因为它涉及到训练一个额外的元模型。
    - 模型多样性：Stacking可以利用不同类型的模型，而加权平均通常用于同质模型的预测结果的组合。
    - 训练过程：Stacking需要一个额外的训练步骤来训练元模型，而加权平均只需要确定权重并计算加权和。
    - 性能提升：Stacking由于考虑了模型之间的相互作用，可能在某些情况下比简单的加权平均提供更好的性能提升。
    - 计算成本：Stacking由于需要训练额外的模型，因此在计算上可能更加昂贵，而加权平均则相对简单且计算成本较低。

在实际应用中，选择哪种融合方法取决于具体问题的需求、可用数据、计算资源以及模型的多样性。有时，简单的加权平均就足够有效，而在需要更复杂模型集成以提高预测准确性的情况下，选择使用Stacking融合可能更好。

代码就不放了.结果从235.2提升到了234.2 我认为应该是属于误差可以忽略不计QaQ. 是时候掏出终极神器了--**神经网络**

## 神经网络
你可以在[lstm_pro.ipynb](https://github.com/machenme/Datawhale/blob/main/power_predict/lstm_pro.ipynb)查看我用`torch`写的LSTM代码,目前成绩大幅提高.从230提高到了1500!虽然是反向提升.估计哪里没写对.
1. 导入必要库和数据
```python
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```
2. 定义数据集
```python
class TimeSeriesDataset(Dataset):
    def __init__(self, df, look_back=100):
        self.look_back = look_back
        self.data, self.labels, self.oot = self.preprocess_data(df)

    def preprocess_data(self, df):
        grouped = df.groupby("id")
        datasets = {id: group.values for id, group in grouped}

        X, Y, OOT = [], [], []
        for id, data in datasets.items():
            for i in range(10, 15):  # 每个id构建5个序列
                a = data[i : (i + self.look_back), 3]
                a = np.append(a, np.array([0] * (self.look_back - len(a))))
                X.append(a[::-1])
                Y.append(data[i - 10 : i, 3][::-1])

            a = data[: self.look_back, 3]
            a = np.append(a, np.array([0] * (self.look_back - len(a))))
            OOT.append(a[::-1])

        return (
            np.array(X, dtype=np.float64),
            np.array(Y, dtype=np.float64),
            np.array(OOT, dtype=np.float64),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(1)
        Y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, Y
```
3. 定义`LSTM`模型
```python
class LSTMModel(nn.Module):
    def __init__(self, look_back, n_features, n_output):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, n_output)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

```
4. 训练模型
```python
# 定义超参数
look_back = 100
n_features = 1
n_output = 10
batch_size = 64
epochs = 10
learning_rate = 0.001

# 创建数据集和数据加载器
train_dataset = TimeSeriesDataset(train, look_back=look_back)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 实例化模型、损失函数和优化器
model = LSTMModel(look_back, n_features, n_output)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
model.train()
for epoch in range(epochs):
    for X, Y in train_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```
5. 模型预测
```python
oot_data = torch.tensor(train_dataset.oot, dtype=torch.float32).unsqueeze(2)
model.eval()
with torch.no_grad():
    predicted_values = model(oot_data).numpy()

```
6. 保存结果
```python
test["target"] = predicted_values.flatten()
test[["id", "dt", "target"]].to_csv("submit_lstm.csv", index=None)
```
</details>
