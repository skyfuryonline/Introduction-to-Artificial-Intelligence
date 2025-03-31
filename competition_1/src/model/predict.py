# from Stacking import StackingClassifier
# from dataprocess.dataset import test_data,data
#
# from prepare_data import prepare_data
#
# # 加载模型
# model = StackingClassifier.load_model("my_stacking_model")
#
# # 准备测试数据
# # 第二个值不需要，因为是预测
# X_test, _ = prepare_data(test_data)
#
# if len(X_test) == 0:
#     print("测试集中没有数据（shot_made_flag全都不为空）")
#
# # 进行预测
# predictions = model.predict(X_test)
# probabilities = model.predict_proba(X_test)
#
# # 将预测结果添加到原始数据
# test_df = X_test.copy()
# test_df['predicted_shot_made'] = predictions
# test_df['shot_made_probability'] = probabilities[:, 1]
#
# # 保存预测结果
# output_path = "predictions.csv"
# test_df.to_csv(output_path, index=False)
# print(f"预测结果已保存至 {output_path}")
#
# # 打印部分结果
# print("\n前5个预测结果：")
# print(test_df[['shot_id', 'predicted_shot_made',
#                'shot_made_probability']].head())



import pandas as pd
from Stacking import StackingClassifier
from dataprocess.dataset import test_data
from prepare_data import prepare_data

# 加载模型
model = StackingClassifier.load_model("my_stacking_model")

# 准备测试数据
X_test, _ = prepare_data(test_data)

# 进行预测
probabilities = model.predict_proba(X_test)

test_df = pd.DataFrame(test_data)
# 确保 shot_id 和 X_test 对应
shot_ids = test_df.loc[X_test.index, 'shot_id'].values

# 生成预测结果 DataFrame
predictions_df = pd.DataFrame({
    'shot_id': shot_ids,
    'shot_made_flag': probabilities[:, 1]  # 仅保留命中概率
})

# 保存为 CSV 文件
output_path = "final_predictions.csv"
predictions_df.to_csv(output_path, index=False)
print(f"预测结果已保存至 {output_path}")