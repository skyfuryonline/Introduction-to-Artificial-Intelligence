import numpy as np

from Stacking import StackingClassifier
# from Stacking import NeuralNetClassifier

from dataprocess.dataset import val_data,train_data
import pandas as pd
from sklearn.metrics import accuracy_score
from prepare_data import prepare_data

# 加载模型
loaded_model = StackingClassifier.load_model("my_stacking_model")
# loaded_model = NeuralNetClassifier.load_model("my_stacking_model")

# RF:查看训练集上的情况：0.9713
X_train,y_train =  prepare_data(train_data)

# RF：使用加载的模型进行预测：最开始：0.5696
X_val,y_val =  prepare_data(val_data,X_train.columns)

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

predictions = loaded_model.predict(X_train)
print(f"训练集准确率: {accuracy_score(y_train, predictions):.4f}")

'''
验证集准确率: 0.6732
训练集准确率: 0.6876

调整参数后：
验证集准确率: 0.6755
训练集准确率: 0.7041

验证集准确率: 0.6770
训练集准确率: 0.6999

下面换用神经网络
验证集准确率: 0.6671
训练集准确率: 0.7198

最后一个是使用复杂堆叠的结果：

'''

# importances = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': loaded_model.rf_model.feature_importances_
# }).sort_values('importance', ascending=False)
# print(importances.head(20))  # 查看前20个重要特征
'''
                              feature  importance
108             action_type_Jump Shot    0.162237
109            action_type_Layup Shot    0.068828
2                       shot_distance    0.057882
6                     distance_period    0.056559
1                               loc_y    0.054519
7                             loc_x_y    0.049984
0                               loc_x    0.047430
4                   seconds_remaining    0.046522
3                   minutes_remaining    0.032348
123     action_type_Running Jump Shot    0.028154
76    shot_zone_basic_Restricted Area    0.023598
129        action_type_Slam Dunk Shot    0.023565
94     action_type_Driving Layup Shot    0.023463
5                              period    0.016183
106        action_type_Jump Bank Shot    0.014698
87      action_type_Driving Dunk Shot    0.014369
111      action_type_Pullup Jump shot    0.012964
82    shot_zone_range_Less Than 8 ft.    0.011838
137  action_type_Turnaround Jump Shot    0.010888
99     action_type_Fadeaway Jump Shot    0.010695
'''
