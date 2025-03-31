import pandas as pd


def prepare_data(dataset, feature_columns=None):
    """
    处理Dataset类型的数据，将其转换为模型可用的特征矩阵X和目标向量y
    参数dataset是datasets库中的Dataset对象
    """
    # 将Dataset转换为DataFrame
    # df = pd.DataFrame(dataset)
    df = dataset.to_pandas()

    # 选择特征
    features = ['loc_x', 'loc_y', 'shot_distance', 'minutes_remaining',
                'seconds_remaining', 'period', 'shot_type', 'shot_zone_area']

    # 处理分类变量
    X  = pd.get_dummies(df[features], columns=['shot_type', 'shot_zone_area'])
    # # 5️⃣ 如果提供 feature_columns，则对齐测试集特征
    # if feature_columns is not None:
    #     missing_cols = set(feature_columns) - set(X.columns)
    #     for col in missing_cols:
    #         X[col] = 0  # 补全缺失特征
    #     X = X[feature_columns]  # 确保列顺序一致

    # X = df  # 特征矩阵

    # 提取目标变量
    y = dataset['shot_made_flag']  # 直接从Dataset中获取目标列

    return X, y