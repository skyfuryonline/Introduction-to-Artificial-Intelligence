import numpy as np

from Stacking import StackingClassifier
from dataprocess.dataset import val_data,train_data
import pandas as pd
from sklearn.metrics import accuracy_score
from prepare_data import prepare_data

# 加载模型
loaded_model = StackingClassifier.load_model("my_stacking_model")

# RF：使用加载的模型进行预测：最开始：0.5696
# X_val,y_val =  prepare_data(val_data)

# RF:查看训练集上的情况：0.9713
X_val,y_val =  prepare_data(train_data)

# print(X_val.columns)
'''
Index(['loc_x', 'loc_y', 'shot_distance', 'minutes_remaining',
       'seconds_remaining', 'period', 'shot_type_2PT Field Goal',
       'shot_type_3PT Field Goal', 'shot_zone_area_Back Court(BC)',
       'shot_zone_area_Center(C)', 'shot_zone_area_Left Side Center(LC)',
       'shot_zone_area_Left Side(L)', 'shot_zone_area_Right Side Center(RC)',
       'shot_zone_area_Right Side(R)'],
      dtype='object')
'''

# print(len(X_val))
# print(len(y_val))
'''
模型已从 my_stacking_model 加载
5140
5140
'''
predictions = loaded_model.predict(X_val)
print(f"验证集准确率: {accuracy_score(y_val, predictions):.4f}")



