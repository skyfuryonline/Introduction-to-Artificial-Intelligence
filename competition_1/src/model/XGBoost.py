from xgboost import XGBClassifier
from config.XGBoostConfig import *

# XGBoost模型
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    objective='binary:logistic',  # 显式指定二分类目标
    base_score=0.5               # 设置合理的基础分数
)