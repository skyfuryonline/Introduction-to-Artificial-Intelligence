# 复杂的stacking

# from Logistic import lr_model
# from XGBoost import xgb_model
# from TabNet import tabnet_model,TabNetClassifier
# from Logistic import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import pickle
# import joblib
# import torch
# from pathlib import Path
#
# # 3. Stacking集成模型
# class StackingClassifier:
#     def __init__(self):
#         self.xgb_model, self.tabnet_model, self.lr_model = xgb_model,tabnet_model,lr_model
#         self.meta_model = LogisticRegression(random_state=42)
#         self.scaler = StandardScaler()
#
#     def fit(self, X, y):
#         # 数据标准化
#
#         X_scaled = self.scaler.fit_transform(X)
#
#         # batchsize=2的时候
#         '''
#         [[-1.  1.  1.  1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.]
#         [ 1. -1. -1. -1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.]]
#
#         [0.0, 0.0]
#         '''
#         # print(X_scaled)
#         # input()
#         # print(y)
#         # input()
#
#         # 训练基础模型并获取预测
#         # XGBoost
#         self.xgb_model.fit(X_scaled, y)
#         xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
#
#         # TabNet
#         self.tabnet_model.fit(
#             X_scaled, y,
#             max_epochs=100,
#             patience=10,
#             batch_size=32,
#             virtual_batch_size=8
#         )
#         tabnet_pred = self.tabnet_model.predict_proba(X_scaled)[:, 1]
#
#         # 逻辑回归
#         self.lr_model.fit(X_scaled, y)
#         lr_pred = self.lr_model.predict_proba(X_scaled)[:, 1]
#
#         # 创建meta特征
#         meta_features = np.column_stack((xgb_pred, tabnet_pred, lr_pred))
#
#         # 训练meta模型
#         self.meta_model.fit(meta_features, y)
#
#         return self
#
#     def predict(self, X):
#         X_scaled = self.scaler.transform(X)
#
#         # 获取基础模型预测
#         xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
#         tabnet_pred = self.tabnet_model.predict_proba(X_scaled)[:, 1]
#         lr_pred = self.lr_model.predict_proba(X_scaled)[:, 1]
#
#         # 创建meta特征
#         meta_features = np.column_stack((xgb_pred, tabnet_pred, lr_pred))
#
#         # meta模型预测
#         return self.meta_model.predict(meta_features)
#
#     def predict_proba(self, X):
#         X_scaled = self.scaler.transform(X)
#
#         # 获取基础模型预测概率
#         xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
#         tabnet_pred = self.tabnet_model.predict_proba(X_scaled)[:, 1]
#         lr_pred = self.lr_model.predict_proba(X_scaled)[:, 1]
#
#         # 创建meta特征
#         meta_features = np.column_stack((xgb_pred, tabnet_pred, lr_pred))
#
#         # meta模型预测概率
#         return self.meta_model.predict_proba(meta_features)
#
#     def save_model(self, directory="saved_model"):
#         """保存整个stacking模型"""
#         # 创建保存目录
#         Path(directory).mkdir(parents=True, exist_ok=True)
#
#         # 1. 保存XGBoost模型
#         joblib.dump(self.xgb_model, f"{directory}/xgb_model.pkl")
#
#         # 2. 保存TabNet模型
#         # torch.save(self.tabnet_model.network.state_dict(),
#         #            f"{directory}/tabnet_model.pt")
#         # # 保存TabNet的配置
#         # tabnet_config = {
#         #     'n_d': self.tabnet_model.n_d,
#         #     'n_a': self.tabnet_model.n_a,
#         #     'n_steps': self.tabnet_model.n_steps,
#         #     'gamma': self.tabnet_model.gamma,
#         #     'n_independent': self.tabnet_model.n_independent,
#         #     'n_shared': self.tabnet_model.n_shared
#         # }
#         # with open(f"{directory}/tabnet_config.pkl", 'wb') as f:
#         #     pickle.dump(tabnet_config, f)
#         self.tabnet_model.save_model(f"{directory}/tabnet_model")
#
#         # 3. 保存逻辑回归模型
#         joblib.dump(self.lr_model, f"{directory}/lr_model.pkl")
#
#         # 4. 保存meta模型
#         joblib.dump(self.meta_model, f"{directory}/meta_model.pkl")
#
#         # 5. 保存scaler
#         joblib.dump(self.scaler, f"{directory}/scaler.pkl")
#
#         print(f"模型已保存至 {directory}")
#
#     @classmethod
#     def load_model(cls, directory="saved_model"):
#         """加载保存的stacking模型"""
#         # 创建新实例
#         model = cls()
#
#         # 1. 加载XGBoost模型
#         model.xgb_model = joblib.load(f"{directory}/xgb_model.pkl")
#
#         # 2. 加载TabNet模型
#         # with open(f"{directory}/tabnet_config.pkl", 'rb') as f:
#         #     tabnet_config = pickle.load(f)
#         # model.tabnet_model = TabNetClassifier(**tabnet_config)
#         # model.tabnet_model.network.load_state_dict(
#         #     torch.load(f"{directory}/tabnet_model.pt")
#         # )
#         # model.tabnet_model.network.eval()
#
#         model.tabnet_model = TabNetClassifier()
#         model.tabnet_model.load_model(f"{directory}/tabnet_model.zip")
#         # model.tabnet_model = TabNetClassifier().load_model(f"{directory}/tabnet_model.zip")
#
#
#         # 3. 加载逻辑回归模型
#         model.lr_model = joblib.load(f"{directory}/lr_model.pkl")
#
#         # 4. 加载meta模型
#         model.meta_model = joblib.load(f"{directory}/meta_model.pkl")
#
#         # 5. 加载scaler
#         model.scaler = joblib.load(f"{directory}/scaler.pkl")
#
#         print(f"模型已从 {directory} 加载")
#         return model


# # 简单的stacking
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import numpy as np
# import joblib
# from pathlib import Path
#
# class StackingClassifier:
#     def __init__(self):
#         self.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#         self.meta_model = LogisticRegression(random_state=42)
#         self.scaler = StandardScaler()
#
#     def fit(self, X, y):
#         # 标准化数据
#         X_scaled = self.scaler.fit_transform(X)
#
#         # 训练 XGBoost
#         self.xgb_model.fit(X_scaled, y)
#         xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
#
#         # 训练元模型（逻辑回归）
#         meta_features = xgb_pred.reshape(-1, 1)
#         self.meta_model.fit(meta_features, y)
#
#     def predict(self, X):
#         X_scaled = self.scaler.transform(X)
#         xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
#         meta_features = xgb_pred.reshape(-1, 1)
#         return self.meta_model.predict(meta_features)
#
#     def predict_proba(self, X):
#         X_scaled = self.scaler.transform(X)
#         xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
#         meta_features = xgb_pred.reshape(-1, 1)
#         return self.meta_model.predict_proba(meta_features)
#
#     def save_model(self, directory="saved_model"):
#         # 创建保存目录
#         Path(directory).mkdir(parents=True, exist_ok=True)
#         joblib.dump(self.xgb_model, f"{directory}/xgb_model.pkl")
#         joblib.dump(self.meta_model, f"{directory}/meta_model.pkl")
#         joblib.dump(self.scaler, f"{directory}/scaler.pkl")
#
#     @classmethod
#     def load_model(cls, directory="saved_model"):
#         model = cls()
#         model.xgb_model = joblib.load(f"{directory}/xgb_model.pkl")
#         model.meta_model = joblib.load(f"{directory}/meta_model.pkl")
#         model.scaler = joblib.load(f"{directory}/scaler.pkl")
#         return model



# # # 过拟合版本RF
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import joblib
# from pathlib import Path
# class StackingClassifier:
#     def __init__(self, n_estimators=800, random_state=42):
#         self.rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
#         self.scaler = StandardScaler()
#
#     def fit(self, X, y):
#         # 数据标准化
#         X_scaled = self.scaler.fit_transform(X)
#         # 训练随机森林
#         self.rf_model.fit(X_scaled, y)
#         return self
#
#     def predict(self, X):
#         X_scaled = self.scaler.transform(X)
#         return self.rf_model.predict(X_scaled)
#
#     def predict_proba(self, X):
#         X_scaled = self.scaler.transform(X)
#         return self.rf_model.predict_proba(X_scaled)
#
#     def save_model(self, directory="saved_rf_model"):
#         Path(directory).mkdir(parents=True, exist_ok=True)
#
#         # 保存随机森林模型
#         joblib.dump(self.rf_model, f"{directory}/rf_model.pkl")
#         # 保存数据标准化器
#         joblib.dump(self.scaler, f"{directory}/scaler.pkl")
#
#         print(f"模型已保存至 {directory}")
#
#     @classmethod
#     def load_model(cls, directory="saved_rf_model"):
#         model = cls()
#         model.rf_model = joblib.load(f"{directory}/rf_model.pkl")
#         model.scaler = joblib.load(f"{directory}/scaler.pkl")
#         print(f"模型已从 {directory} 加载")
#         return model


# # 调参版
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from pathlib import Path

class StackingClassifier:
    def __init__(self, n_estimators=800, max_depth=20, min_samples_leaf=5, random_state=42):
        """
        初始化随机森林分类器，添加控制过拟合的参数
        """
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,          # 限制树的最大深度
            min_samples_leaf=min_samples_leaf,  # 限制叶子节点的最小样本数
            random_state=random_state
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        训练模型：标准化数据并拟合随机森林
        """
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        # 训练随机森林
        self.rf_model.fit(X_scaled, y)

        # # 打印训练集准确率
        # train_accuracy = self.rf_model.score(X_scaled, y)
        # print(f"训练集准确率: {train_accuracy:.4f}")
        return self

    def predict(self, X):
        """
        预测类别
        """
        X_scaled = self.scaler.transform(X)
        return self.rf_model.predict(X_scaled)

    def predict_proba(self, X):
        """
        预测概率
        """
        X_scaled = self.scaler.transform(X)
        return self.rf_model.predict_proba(X_scaled)

    def score(self, X, y):
        """
        计算准确率
        """
        X_scaled = self.scaler.transform(X)
        return self.rf_model.score(X_scaled, y)

    def save_model(self, directory="saved_rf_model"):
        """
        保存模型和标准化器
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.rf_model, f"{directory}/rf_model.pkl")
        joblib.dump(self.scaler, f"{directory}/scaler.pkl")
        print(f"模型已保存至 {directory}")

    @classmethod
    def load_model(cls, directory="saved_rf_model"):
        """
        加载模型和标准化器
        """
        model = cls()
        model.rf_model = joblib.load(f"{directory}/rf_model.pkl")
        model.scaler = joblib.load(f"{directory}/scaler.pkl")
        print(f"模型已从 {directory} 加载")
        return model



# 换用XGBoost
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import VotingClassifier
# import joblib
# from pathlib import Path
#
# class StackingClassifier:
#     def __init__(self, n_estimators=300, max_depth=4, learning_rate=0.1, reg_lambda=1.0, random_state=42):
#         """
#         初始化集成模型：XGBoost + LightGBM
#         """
#         self.xgb = XGBClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             learning_rate=learning_rate,
#             reg_lambda=reg_lambda,
#             random_state=random_state,
#             eval_metric='logloss',
#             use_label_encoder=False
#         )
#         self.lgbm = LGBMClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             learning_rate=learning_rate,
#             reg_lambda=reg_lambda,
#             random_state=random_state
#         )
#         self.model = VotingClassifier(
#             estimators=[('xgb', self.xgb), ('lgbm', self.lgbm)],
#             voting='soft'
#         )
#         self.scaler = StandardScaler()
#
#     def fit(self, X, y):
#         """训练模型"""
#         X_scaled = self.scaler.fit_transform(X)
#         self.model.fit(X_scaled, y)
#         train_accuracy = self.model.score(X_scaled, y)
#         print(f"训练集准确率: {train_accuracy:.4f}")
#         return self
#
#     def predict(self, X):
#         """预测类别"""
#         X_scaled = self.scaler.transform(X)
#         return self.model.predict(X_scaled)
#
#     def predict_proba(self, X):
#         """预测概率"""
#         X_scaled = self.scaler.transform(X)
#         return self.model.predict_proba(X_scaled)
#
#     def score(self, X, y):
#         """计算准确率"""
#         X_scaled = self.scaler.transform(X)
#         return self.model.score(X_scaled, y)
#
#     def save_model(self, directory="saved_voting_model"):
#         """保存模型"""
#         Path(directory).mkdir(parents=True, exist_ok=True)
#         joblib.dump(self.model, f"{directory}/voting_model.pkl")
#         joblib.dump(self.scaler, f"{directory}/scaler.pkl")
#         print(f"模型已保存至 {directory}")
#
#     @classmethod
#     def load_model(cls, directory="saved_voting_model"):
#         """加载模型"""
#         model = cls()
#         model.model = joblib.load(f"{directory}/voting_model.pkl")
#         model.scaler = joblib.load(f"{directory}/scaler.pkl")
#         print(f"模型已从 {directory} 加载")
#         return model


# 神经网络
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# import joblib
# from pathlib import Path
#
#
# class NeuralNetClassifier:
#     def __init__(self, hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42):
#         """
#         初始化神经网络分类器
#         """
#         self.model = MLPClassifier(
#             hidden_layer_sizes=hidden_layer_sizes,  # 两层隐藏层：100和50个神经元
#             activation='relu',  # ReLU激活函数
#             solver='adam',  # Adam优化器
#             max_iter=max_iter,
#             random_state=random_state,
#             early_stopping=True,  # 早停防止过拟合
#             validation_fraction=0.1,  # 10%数据用于验证
#             n_iter_no_change=10  # 早停耐心
#         )
#         self.scaler = StandardScaler()
#
#     def fit(self, X, y):
#         """训练模型"""
#         X_scaled = self.scaler.fit_transform(X)
#         self.model.fit(X_scaled, y)
#         train_accuracy = self.model.score(X_scaled, y)
#         print(f"训练集准确率: {train_accuracy:.4f}")
#
#         # 交叉验证
#         cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
#         print(f"5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
#         return self
#
#     def predict(self, X):
#         """预测类别"""
#         X_scaled = self.scaler.transform(X)
#         return self.model.predict(X_scaled)
#
#     def predict_proba(self, X):
#         """预测概率"""
#         X_scaled = self.scaler.transform(X)
#         return self.model.predict_proba(X_scaled)
#
#     def score(self, X, y):
#         """计算准确率"""
#         X_scaled = self.scaler.transform(X)
#         return self.model.score(X_scaled, y)
#
#     def save_model(self, directory="saved_mlp_model"):
#         """保存模型"""
#         Path(directory).mkdir(parents=True, exist_ok=True)
#         joblib.dump(self.model, f"{directory}/mlp_model.pkl")
#         joblib.dump(self.scaler, f"{directory}/scaler.pkl")
#         print(f"模型已保存至 {directory}")
#
#     @classmethod
#     def load_model(cls, directory="saved_mlp_model"):
#         """加载模型"""
#         model = cls()
#         model.model = joblib.load(f"{directory}/mlp_model.pkl")
#         model.scaler = joblib.load(f"{directory}/scaler.pkl")
#         print(f"模型已从 {directory} 加载")
#         return model



# XGBoost结合网格搜索
# from xgboost import XGBClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# import joblib
# from pathlib import Path
# import numpy as np
# import pandas as pd
# class StackingClassifier:
#     def __init__(self, n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42):
#         """
#         初始化XGBoost分类器
#         """
#         self.model = XGBClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             learning_rate=learning_rate,
#             random_state=random_state,
#             eval_metric='logloss',
#             use_label_encoder=False
#         )
#         self.scaler = StandardScaler()
#
#     def fit(self, X, y):
#         """训练模型"""
#         X_scaled = self.scaler.fit_transform(X)
#         self.model.fit(X_scaled, y)
#         train_accuracy = self.model.score(X_scaled, y)
#         print(f"训练集准确率: {train_accuracy:.4f}")
#
#         # 交叉验证
#         cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
#         print(f"5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
#         return self
#
#     def predict(self, X):
#         """预测类别"""
#         X_scaled = self.scaler.transform(X)
#         return self.model.predict(X_scaled)
#
#     def predict_proba(self, X):
#         """预测概率"""
#         X_scaled = self.scaler.transform(X)
#         return self.model.predict_proba(X_scaled)
#
#     def score(self, X, y):
#         """计算准确率"""
#         X_scaled = self.scaler.transform(X)
#         return self.model.score(X_scaled, y)
#
#     def feature_importance(self, feature_names):
#         """输出特征重要性"""
#         importances = pd.DataFrame({
#             'feature': feature_names,
#             'importance': self.model.feature_importances_
#         }).sort_values('importance', ascending=False)
#         print(importances.head(20))
#         return importances
#
#     def save_model(self, directory="saved_xgb_model"):
#         """保存模型"""
#         Path(directory).mkdir(parents=True, exist_ok=True)
#         joblib.dump(self.model, f"{directory}/xgb_model.pkl")
#         joblib.dump(self.scaler, f"{directory}/scaler.pkl")
#         print(f"模型已保存至 {directory}")
#
#     @classmethod
#     def load_model(cls, directory="saved_xgb_model"):
#         """加载模型"""
#         model = cls()
#         model.model = joblib.load(f"{directory}/xgb_model.pkl")
#         model.scaler = joblib.load(f"{directory}/scaler.pkl")
#         print(f"模型已从 {directory} 加载")
#         return model