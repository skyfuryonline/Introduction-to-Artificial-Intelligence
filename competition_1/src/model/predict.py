from Stacking import StackingClassifier
from dataprocess.dataset import test_data

from prepare_data import prepare_data



# 加载模型
model = StackingClassifier.load_model("my_stacking_model")

# 准备测试数据
# 第二个值不需要，因为是预测
X_test, _ = prepare_data(test_data)

if len(X_test) == 0:
    print("测试集中没有数据（shot_made_flag全都不为空）")

# 进行预测
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 将预测结果添加到原始数据
test_df = X_test.copy()
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