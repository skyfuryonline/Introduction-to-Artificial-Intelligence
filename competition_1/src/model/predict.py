from Stacking import StackingClassifier
from dataprocess.dataset import test_data
import pandas as pd

# 加载模型
model = StackingClassifier.load_model("my_stacking_model")

# 准备测试数据
test_df = pd.DataFrame(test_data)

# 选择特征并处理分类变量
features = ['loc_x', 'loc_y', 'shot_distance', 'minutes_remaining',
            'seconds_remaining', 'period', 'shot_type', 'shot_zone_area']
X_test = pd.get_dummies(test_df[features],
                        columns=['shot_type', 'shot_zone_area'])

if len(X_test) == 0:
    print("测试集中没有数据（shot_made_flag全都不为空）")

# 进行预测
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 将预测结果添加到原始数据
test_df = test_df.copy()
test_df['predicted_shot_made'] = predictions
test_df['shot_made_probability'] = probabilities[:, 1]

# 保存预测结果
output_path = "predictions.csv"
test_df.to_csv(output_path, index=False)
print(f"预测结果已保存至 {output_path}")

# 打印部分结果
print("\n前5个预测结果：")
print(test_df[['shot_id', 'predicted_shot_made',
               'shot_made_probability']].head())