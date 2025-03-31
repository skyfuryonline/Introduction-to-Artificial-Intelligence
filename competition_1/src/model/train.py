from Stacking import StackingClassifier
from dataprocess.dataset import train_data,val_data
import pandas as pd

def prepare_data(sample):
    # 将你的样本转换为DataFrame
    df = pd.DataFrame([sample])

    # 选择特征
    features = ['loc_x', 'loc_y', 'shot_distance', 'minutes_remaining',
                'seconds_remaining', 'period', 'shot_type', 'shot_zone_area']

    # 处理分类变量
    df = pd.get_dummies(df[features], columns=['shot_type', 'shot_zone_area'])
    X = df
    y = sample['shot_made_flag']
    return X, y


model = StackingClassifier()

X,y = prepare_data(train_data)

model.fit(X,y)
# 保存模型
model.save_model("my_stacking_model")
