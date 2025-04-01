import pandas as pd
from Stacking import StackingClassifier
# from Stacking import NeuralNetClassifier

from dataprocess.dataset import train_data,val_data
from prepare_data import prepare_data

model = StackingClassifier()
# model = NeuralNetClassifier()

# # 计算 'shot_made_flag' 列的取值占比
# train_data = pd.DataFrame(train_data)
# value_counts = train_data['shot_made_flag'].value_counts(normalize=True)
# # 输出占比
# print('标签占比为：',value_counts)
'''
标签占比为： shot_made_flag
0.0    0.555139
1.0    0.444861
'''

X,y = prepare_data(train_data)
# print(X.columns)
'''
Index(['loc_x', 'loc_y', 'shot_distance', 'minutes_remaining',
       'seconds_remaining', 'period', 'shot_type_2PT Field Goal',
       'shot_type_3PT Field Goal', 'shot_zone_area_Back Court(BC)',
       'shot_zone_area_Center(C)', 'shot_zone_area_Left Side Center(LC)',
       'shot_zone_area_Left Side(L)', 'shot_zone_area_Right Side Center(RC)',
       'shot_zone_area_Right Side(R)'],
      dtype='object')
'''
print("开始训练！")

# 顺序切分
# X = X.iloc[:2000]
# y = y[:2000]  # y是列表，直接切片

model.fit(X,y)

# 保存模型
print("开始保存模型！")
model.save_model("best_model")


# XGBoost使用网格搜索
# from xgboost import XGBClassifier
# model = StackingClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
# # 准备数据
# X_train, y_train = prepare_data(train_data)
# X_val, y_val = prepare_data(val_data, feature_columns=X_train.columns)
# # 网格搜索优化超参数
# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'n_estimators': [300, 500, 700],
#     'max_depth': [4, 6, 8],
#     'learning_rate': [0.01, 0.05, 0.1]
# }
# xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
# x_train_scaled = model.scaler.fit_transform(X_train)
# grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(x_train_scaled, y_train)
# print("最佳参数:", grid_search.best_params_)
# print("最佳交叉验证准确率:", grid_search.best_score_)
#
# # 使用最佳参数重新训练
# best_model = StackingClassifier(
#     n_estimators=grid_search.best_params_['n_estimators'],
#     max_depth=grid_search.best_params_['max_depth'],
#     learning_rate=grid_search.best_params_['learning_rate'],
#     random_state=42
# )
# best_model.fit(X_train, y_train)
# final_val_accuracy = best_model.score(X_val, y_val)
# print(f"最终验证集准确率: {final_val_accuracy:.4f}")
#
# # 保存最佳模型
# best_model.save_model("best_xgb_model")




# #stacking使用网格搜索
#一个参数组合完成5折交叉验证的时间每次24分钟，总运行时间约为 32小时24分钟。

# # 直接使用 StackingClassifier
# from sklearn.model_selection import GridSearchCV
#
# # 定义参数网格
# param_grid = {
#     'xgb_params': [
#         {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'reg_lambda': 1.0},
#         {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'reg_lambda': 2.0},
#         {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.01, 'reg_lambda': 0.5}
#     ],
#     'tabnet_params': [
#         {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3},
#         {'n_d': 16, 'n_a': 16, 'n_steps': 5, 'gamma': 1.5},
#         {'n_d': 32, 'n_a': 32, 'n_steps': 7, 'gamma': 1.7}
#     ],
#     'lr_params': [
#         {'C': 0.1, 'max_iter': 1000},
#         {'C': 1.0, 'max_iter': 1000},
#         {'C': 10.0, 'max_iter': 1000}
#     ],
#     'meta_params': [
#         {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 5},
#         {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 5},
#         {'n_estimators': 200, 'max_depth': 15, 'min_samples_leaf': 3}
#     ],
#     'meta_model_type': ['rf']
# }
#
# # 创建网格搜索对象
# stacking_model = StackingClassifier(random_state=42)
# grid_search = GridSearchCV(
#     estimator=stacking_model,
#     param_grid=param_grid,
#     cv=5,# 指定使用5折交叉验证来评估模型性能。数据集会被自动划分为 5 个子集（folds）。每次训练时，4 个子集用于训练，1 个子集用于验证。
#     scoring='accuracy',
#     n_jobs=1,  # TabNet不支持多进程
#     verbose=2
# )
#
# # 假设 X_train, y_train, X_val, y_val 已准备好
# grid_search.fit(X, y)
#
# # 输出结果
# print("最佳参数:", grid_search.best_params_)
# print("最佳交叉验证准确率:", grid_search.best_score_)
#
# # 使用最佳模型评估验证集
# best_model = grid_search.best_estimator_
#
# # 保存最佳模型
# best_model.save_model("best_stacking_model")