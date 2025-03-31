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



# # 新增更多的特征
# import pandas as pd
#
# def prepare_data(dataset, feature_columns=None):
#     """
#     处理Dataset类型的数据，将其转换为模型可用的特征矩阵X和目标向量y
#     参数dataset是datasets库中的Dataset对象
#     """
#     # 将Dataset转换为DataFrame
#     df = dataset.to_pandas()
#
#     # 扩展特征列表
#     features = [
#         'loc_x', 'loc_y', 'shot_distance', 'minutes_remaining', 'seconds_remaining', 'period',  # 数值特征
#         'shot_type', 'shot_zone_area', 'season', 'opponent', 'playoffs', 'shot_zone_basic',
#         'shot_zone_range', 'action_type'  # 分类特征
#     ]
#
#     # 选择特征并处理分类变量
#     X = pd.get_dummies(df[features],
#                       columns=['shot_type', 'shot_zone_area', 'season', 'opponent',
#                                'playoffs', 'shot_zone_basic', 'shot_zone_range', 'action_type'])
#
#     # 如果提供 feature_columns，则对齐测试集特征
#     if feature_columns is not None:
#         missing_cols = set(feature_columns) - set(X.columns)
#         for col in missing_cols:
#             X[col] = 0  # 补全缺失特征
#         X = X[feature_columns]  # 确保列顺序一致
#
#     # 提取目标变量
#     y = dataset['shot_made_flag']
#
#     return X, y


# # 新增交互项
# def prepare_data(dataset, feature_columns=None):
#     df = dataset.to_pandas()
#     features = [
#         'loc_x', 'loc_y', 'shot_distance', 'minutes_remaining', 'seconds_remaining', 'period',
#         'shot_type', 'shot_zone_area', 'season', 'opponent', 'playoffs', 'shot_zone_basic',
#         'shot_zone_range', 'action_type'
#     ]
#     # 添加交互特征
#     df['distance_period'] = df['shot_distance'] * df['period']
#     df['loc_x_y'] = df['loc_x'] * df['loc_y']
#     features.extend(['distance_period', 'loc_x_y'])
#
#     X = pd.get_dummies(df[features],
#                        columns=['shot_type', 'shot_zone_area', 'season', 'opponent',
#                                 'playoffs', 'shot_zone_basic', 'shot_zone_range', 'action_type'])
#     if feature_columns is not None:
#         missing_cols = set(feature_columns) - set(X.columns)
#         for col in missing_cols:
#             X[col] = 0
#         X = X[feature_columns]
#     y = dataset['shot_made_flag']
#     return X, y


# 进一步改进交互项
# import pandas as pd
# def prepare_data(dataset, feature_columns=None):
#     """
#     处理Dataset类型的数据，生成适合神经网络的特征矩阵X和目标向量y
#     """
#     df = dataset.to_pandas()
#
#     # 使用所有可能有用的特征
#     features = [
#         'loc_x', 'loc_y', 'shot_distance', 'minutes_remaining', 'seconds_remaining', 'period',  # 数值
#         'action_type', 'shot_zone_basic', 'shot_zone_range', 'shot_type', 'shot_zone_area',
#         'season', 'opponent', 'playoffs'  # 分类
#     ]
#
#     # 添加衍生特征和交互项
#     df['time_remaining'] = df['minutes_remaining'] * 60 + df['seconds_remaining']
#     df['distance_action'] = df['shot_distance'] * pd.factorize(df['action_type'])[0]
#     df['loc_y_action'] = df['loc_y'] * pd.factorize(df['action_type'])[0]
#     df['distance_zone'] = df['shot_distance'] * pd.factorize(df['shot_zone_basic'])[0]
#     df['is_close'] = (df['shot_distance'] < 8).astype(int)
#     df['is_playoff_close'] = df['playoffs'] * df['is_close']
#
#     derived_features = ['time_remaining', 'distance_action', 'loc_y_action', 'distance_zone',
#                        'is_close', 'is_playoff_close']
#     features.extend(derived_features)
#
#     # 处理分类变量
#     categorical_cols = [
#         'action_type', 'shot_zone_basic', 'shot_zone_range', 'shot_type', 'shot_zone_area',
#         'season', 'opponent', 'playoffs'
#     ]
#     X = pd.get_dummies(df[features], columns=categorical_cols)
#
#     # 对齐特征
#     if feature_columns is not None:
#         missing_cols = set(feature_columns) - set(X.columns)
#         for col in missing_cols:
#             X[col] = 0
#         X = X[feature_columns]
#
#     y = dataset['shot_made_flag']
#     return X, y