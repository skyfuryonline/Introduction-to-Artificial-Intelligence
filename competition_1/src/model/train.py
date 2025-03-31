from Stacking import StackingClassifier
from dataprocess.dataset import train_data
from prepare_data import prepare_data

model = StackingClassifier()

X,y = prepare_data(train_data)

model.fit(X,y)
# 保存模型
model.save_model("my_stacking_model")
