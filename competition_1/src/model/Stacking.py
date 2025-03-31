from Logistic import lr_model
from XGBoost import xgb_model
from TabNet import tabnet_model
from Logistic import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

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
