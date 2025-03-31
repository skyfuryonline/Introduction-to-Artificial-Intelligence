from Logistic import lr_model
from XGBoost import xgb_model
from TabNet import tabnet_model,TabNetClassifier
from Logistic import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import joblib
import torch
from pathlib import Path

# 3. Stacking集成模型
class StackingClassifier:
    def __init__(self):
        self.xgb_model, self.tabnet_model, self.lr_model = xgb_model,tabnet_model,lr_model
        self.meta_model = LogisticRegression(random_state=42)

        self.scaler = StandardScaler()

    def fit(self, X, y):
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练基础模型并获取预测
        # XGBoost
        self.xgb_model.fit(X_scaled, y)
        xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]

        # TabNet
        self.tabnet_model.fit(
            X_scaled, y,
            max_epochs=100,
            patience=10,
            batch_size=32,
            virtual_batch_size=8
        )
        tabnet_pred = self.tabnet_model.predict_proba(X_scaled)[:, 1]

        # 逻辑回归
        self.lr_model.fit(X_scaled, y)
        lr_pred = self.lr_model.predict_proba(X_scaled)[:, 1]

        # 创建meta特征
        meta_features = np.column_stack((xgb_pred, tabnet_pred, lr_pred))

        # 训练meta模型
        self.meta_model.fit(meta_features, y)

        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)

        # 获取基础模型预测
        xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
        tabnet_pred = self.tabnet_model.predict_proba(X_scaled)[:, 1]
        lr_pred = self.lr_model.predict_proba(X_scaled)[:, 1]

        # 创建meta特征
        meta_features = np.column_stack((xgb_pred, tabnet_pred, lr_pred))

        # meta模型预测
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)

        # 获取基础模型预测概率
        xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
        tabnet_pred = self.tabnet_model.predict_proba(X_scaled)[:, 1]
        lr_pred = self.lr_model.predict_proba(X_scaled)[:, 1]

        # 创建meta特征
        meta_features = np.column_stack((xgb_pred, tabnet_pred, lr_pred))

        # meta模型预测概率
        return self.meta_model.predict_proba(meta_features)

    def save_model(self, directory="saved_model"):
        """保存整个stacking模型"""
        # 创建保存目录
        Path(directory).mkdir(parents=True, exist_ok=True)

        # 1. 保存XGBoost模型
        joblib.dump(self.xgb_model, f"{directory}/xgb_model.pkl")

        # 2. 保存TabNet模型
        torch.save(self.tabnet_model.network.state_dict(),
                   f"{directory}/tabnet_model.pt")
        # 保存TabNet的配置
        tabnet_config = {
            'n_d': self.tabnet_model.n_d,
            'n_a': self.tabnet_model.n_a,
            'n_steps': self.tabnet_model.n_steps,
            'gamma': self.tabnet_model.gamma,
            'n_independent': self.tabnet_model.n_independent,
            'n_shared': self.tabnet_model.n_shared
        }
        with open(f"{directory}/tabnet_config.pkl", 'wb') as f:
            pickle.dump(tabnet_config, f)

        # 3. 保存逻辑回归模型
        joblib.dump(self.lr_model, f"{directory}/lr_model.pkl")

        # 4. 保存meta模型
        joblib.dump(self.meta_model, f"{directory}/meta_model.pkl")

        # 5. 保存scaler
        joblib.dump(self.scaler, f"{directory}/scaler.pkl")

        print(f"模型已保存至 {directory}")

    @classmethod
    def load_model(cls, directory="saved_model"):
        """加载保存的stacking模型"""
        # 创建新实例
        model = cls()

        # 1. 加载XGBoost模型
        model.xgb_model = joblib.load(f"{directory}/xgb_model.pkl")

        # 2. 加载TabNet模型
        with open(f"{directory}/tabnet_config.pkl", 'rb') as f:
            tabnet_config = pickle.load(f)
        model.tabnet_model = TabNetClassifier(**tabnet_config)
        model.tabnet_model.network.load_state_dict(
            torch.load(f"{directory}/tabnet_model.pt")
        )
        model.tabnet_model.network.eval()

        # 3. 加载逻辑回归模型
        model.lr_model = joblib.load(f"{directory}/lr_model.pkl")

        # 4. 加载meta模型
        model.meta_model = joblib.load(f"{directory}/meta_model.pkl")

        # 5. 加载scaler
        model.scaler = joblib.load(f"{directory}/scaler.pkl")

        print(f"模型已从 {directory} 加载")
        return model
