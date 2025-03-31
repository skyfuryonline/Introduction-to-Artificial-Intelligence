from Stacking import StackingClassifier
from dataprocess.dataset import val_data
import pandas as pd
from sklearn.metrics import accuracy_score

# 加载模型
loaded_model = StackingClassifier.load_model("my_stacking_model")

# 使用加载的模型进行预测
X_val = pd.get_dummies(pd.DataFrame(val_data))
y_val = X_val.pop('shot_made_flag')
predictions = loaded_model.predict(X_val)
print(f"验证集准确率: {accuracy_score(y_val, predictions):.4f}")


