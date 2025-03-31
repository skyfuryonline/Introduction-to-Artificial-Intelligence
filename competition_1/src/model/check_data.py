from dataprocess.dataset import val_data,train_data
import pandas as pd
import numpy as np
from scipy import stats
from datasets import Dataset

def compare_distributions(train_dataset, val_dataset, features):
    # 将Dataset转换为DataFrame
    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(val_dataset)

    print("=== 数据分布比较 ===")
    print(f"训练集样本量: {len(train_df)}")
    print(f"验证集样本量: {len(val_df)}")

    print("\n1. 数值特征统计:")
    numeric_features = ['loc_x', 'loc_y', 'shot_distance', 'minutes_remaining',
                        'seconds_remaining', 'period']

    for feature in numeric_features:
        train_stats = train_df[feature].describe()
        val_stats = val_df[feature].describe()

        # 进行t检验
        t_stat, p_value = stats.ttest_ind(train_df[feature], val_df[feature],
                                          equal_var=False)  # 使用Welch's t-test

        print(f"\n{feature}:")
        print("训练集统计:")
        print(train_stats.round(2))
        print("验证集统计:")
        print(val_stats.round(2))
        print(f"t检验 p值: {p_value:.4f}")
        print(f"分布差异显著性: {'显著' if p_value < 0.05 else '不显著'}")

    print("\n2. 分类特征分布:")
    categorical_features = ['shot_type', 'shot_zone_area']

    for feature in categorical_features:
        print(f"\n{feature}:")
        train_dist = train_df[feature].value_counts(normalize=True)
        val_dist = val_df[feature].value_counts(normalize=True)

        print("训练集分布 (前5):")
        print(train_dist.head().round(3))
        print("验证集分布 (前5):")
        print(val_dist.head().round(3))

        # 创建列联表进行卡方检验
        all_categories = set(train_dist.index).union(set(val_dist.index))
        contingency_table = pd.DataFrame({
            'train': [train_dist.get(cat, 0) * len(train_df) for cat in all_categories],
            'val': [val_dist.get(cat, 0) * len(val_df) for cat in all_categories]
        }, index=list(all_categories))

        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        print(f"卡方检验 p值: {p_value:.4f}")
        print(f"分布差异显著性: {'显著' if p_value < 0.05 else '不显著'}")

    print("\n3. 目标变量分布:")
    train_target = train_df['shot_made_flag'].value_counts(normalize=True)
    val_target = val_df['shot_made_flag'].value_counts(normalize=True)
    print("训练集目标分布:")
    print(train_target.round(3))
    print("验证集目标分布:")
    print(val_target.round(3))

    # 计算目标分布的卡方检验
    contingency_target = pd.DataFrame({
        'train': train_target * len(train_df),
        'val': val_target.reindex(train_target.index, fill_value=0) * len(val_df)
    })
    chi2_target, p_target, _, _ = stats.chi2_contingency(contingency_target)
    print(f"目标分布卡方检验 p值: {p_target:.4f}")
    print(f"目标分布差异显著性: {'显著' if p_target < 0.05 else '不显著'}")


# 选择的特征
features = ['loc_x', 'loc_y', 'shot_distance', 'minutes_remaining',
            'seconds_remaining', 'period', 'shot_type', 'shot_zone_area']

# 执行分布比较
# 假设你的数据集变量名为train_dataset和val_dataset
# compare_distributions(train_dataset, val_dataset, features)

# 如果你想测试代码，可以先用小样本运行：
small_train = train_data
small_val = val_data
compare_distributions(small_train, small_val, features)