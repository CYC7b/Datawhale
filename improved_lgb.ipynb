import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# 加载数据
# 假设 data 是包含特征和标签的 DataFrame
X = data.drop(columns=['label'])
y = data['label']

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义特征工程函数
def feature_engineering(df):
    # 添加新的特征
    df['new_feature'] = df['feature1'] * df['feature2']  # 示例
    return df

# 应用特征工程
X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)

# 定义LightGBM参数
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "max_depth": 7,
    "learning_rate": 0.02,
    "verbose": -1,
}

# 网格搜索参数调优
param_grid = {
    'num_leaves': [31, 50],
    'learning_rate': [0.01, 0.02],
    'n_estimators': [100, 200],
    'max_depth': [7, 9]
}

# 定义模型
lgb_estimator = lgb.LGBMRegressor(**params)

# 交叉验证
gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
gsearch.fit(X_train, y_train)

# 获取最优参数
best_params = gsearch.best_params_

# 使用最优参数重新训练模型
best_model = lgb.LGBMRegressor(**best_params)
best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=100)

# 预测
y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Test RMSE: {rmse}")

# 结果保存
df_submit = pd.DataFrame({'id': test_ids, 'mRNA_remaining_pct': y_pred})
df_submit.to_csv("submission.csv", index=False)

# 集成学习 (可选)
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# 定义不同的回归模型
xgb_model = XGBRegressor()
rf_model = RandomForestRegressor()

# 集成模型
ensemble_model = VotingRegressor(estimators=[
    ('lgb', best_model),
    ('xgb', xgb_model),
    ('rf', rf_model)
])

# 训练集成模型
ensemble_model.fit(X_train, y_train)

# 预测
ensemble_pred = ensemble_model.predict(X_test)
ensemble_rmse = mean_squared_error(y_test, ensemble_pred, squared=False)
print(f"Ensemble Test RMSE: {ensemble_rmse}")

# 保存结果
df_ensemble_submit = pd.DataFrame({'id': test_ids, 'mRNA_remaining_pct': ensemble_pred})
df_ensemble_submit.to_csv("ensemble_submission.csv", index=False)
