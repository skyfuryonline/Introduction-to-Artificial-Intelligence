from xgboost import XGBClassifier
from config.XGBoostConfig import *

# XGBoost模型
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)