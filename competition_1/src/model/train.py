import pandas as pd

from Stacking import StackingClassifier
from dataprocess.dataset import train_data
from prepare_data import prepare_data

model = StackingClassifier()


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
model.save_model("my_stacking_model")
