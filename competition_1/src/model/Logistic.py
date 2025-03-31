from sklearn.linear_model import LogisticRegression
from config.LogisticConfig import *

# 逻辑回归模型
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)