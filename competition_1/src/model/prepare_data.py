import pandas as pd
def prepare_data(dataset, feature_columns=None):
    """
    处理Dataset类型的数据，将其转换为模型可用的特征矩阵X和目标向量y
    参数dataset是datasets库中的Dataset对象
    """
    df = dataset.to_pandas()
    # 选择特征
    features = ['loc_x', 'loc_y', 'shot_distance', 'minutes_remaining',
                'seconds_remaining', 'period', 'shot_type', 'shot_zone_area']
    # 处理分类变量
    X  = pd.get_dummies(df[features], columns=['shot_type', 'shot_zone_area'])
    # 5️⃣ 如果提供 feature_columns，则对齐测试集特征
    if feature_columns is not None:
        missing_cols = set(feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0  # 补全缺失特征
        X = X[feature_columns]  # 确保列顺序一致
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


# # 进一步改进交互项
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


# 再改进数据交互--目前次好
# import pandas as pd
# import numpy as np
#
# def prepare_data(dataset, feature_columns=None):
#     """
#     处理Dataset类型的数据，生成适合神经网络的丰富特征矩阵X和目标向量y
#     """
#     df = dataset.to_pandas()
#
#     # 使用所有可能有用的特征
#     features = [
#         'loc_x', 'loc_y', 'shot_distance', 'minutes_remaining', 'seconds_remaining', 'period',  # 数值
#         'lat', 'lon', 'team_id',  # 新增原始特征
#         'action_type', 'shot_zone_basic', 'shot_zone_range', 'shot_type', 'shot_zone_area',
#         'season', 'opponent', 'playoffs'  # 分类
#     ]
#
#     # 添加衍生特征
#     df['time_remaining'] = df['minutes_remaining'] * 60 + df['seconds_remaining']  # 总剩余秒数
#     df['distance_squared'] = df['shot_distance'] ** 2  # 距离平方
#     df['shot_angle'] = np.arctan2(df['loc_y'], df['loc_x'].replace(0, 1e-6))  # 投篮角度（避免除以0）
#     df['is_close'] = (df['shot_distance'] < 8).astype(int)  # 是否近距离投篮
#     df['is_end_of_period'] = (df['time_remaining'] < 10).astype(int)  # 是否节末投篮
#     df['year'] = pd.to_datetime(df['game_date']).dt.year  # 从game_date提取年份
#
#     # 添加交互项
#     df['distance_action'] = df['shot_distance'] * pd.factorize(df['action_type'])[0]
#     df['loc_y_action'] = df['loc_y'] * pd.factorize(df['action_type'])[0]
#     df['loc_x_action'] = df['loc_x'] * pd.factorize(df['action_type'])[0]
#     df['distance_zone'] = df['shot_distance'] * pd.factorize(df['shot_zone_basic'])[0]
#     df['time_action'] = df['time_remaining'] * pd.factorize(df['action_type'])[0]
#     df['distance_opponent'] = df['shot_distance'] * pd.factorize(df['opponent'])[0]
#     df['is_playoff_close'] = df['playoffs'] * df['is_close']
#
#     # 扩展特征列表
#     derived_features = [
#         'time_remaining', 'distance_squared', 'shot_angle', 'is_close', 'is_end_of_period', 'year',
#         'distance_action', 'loc_y_action', 'loc_x_action', 'distance_zone', 'time_action',
#         'distance_opponent', 'is_playoff_close'
#     ]
#     features.extend(derived_features)
#
#     # 处理分类变量
#     categorical_cols = [
#         'action_type', 'shot_zone_basic', 'shot_zone_range', 'shot_type', 'shot_zone_area',
#         'season', 'opponent', 'playoffs', 'team_id'
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


# 究极版数据交互，效果似乎不如上面
# import pandas as pd
# import numpy as np
#
# def prepare_data(dataset, feature_columns=None):
#     """
#     处理Dataset类型的数据，生成适合神经网络的丰富特征矩阵X和目标向量y
#     """
#     df = dataset.to_pandas()
#
#     # 使用所有可能有用的特征
#     features = [
#         'loc_x', 'loc_y', 'shot_distance', 'minutes_remaining', 'seconds_remaining', 'period',  # 数值
#         'lat', 'lon', 'team_id',  # 新增原始特征
#         'action_type', 'shot_zone_basic', 'shot_zone_range', 'shot_type', 'shot_zone_area',
#         'season', 'opponent', 'playoffs'  # 分类
#     ]
#
#     # 添加衍生特征
#     df['time_remaining'] = df['minutes_remaining'] * 60 + df['seconds_remaining']  # 总剩余秒数
#     df['distance_squared'] = df['shot_distance'] ** 2  # 距离平方
#     df['distance_cubed'] = df['shot_distance'] ** 3  # 距离立方
#     df['distance_log'] = np.log1p(df['shot_distance'])  # 距离对数（避免log(0)）
#     df['shot_angle'] = np.arctan2(df['loc_y'], df['loc_x'].replace(0, 1e-6))  # 投篮角度
#     df['shot_angle_sin'] = np.sin(df['shot_angle'])  # 角度正弦
#     df['shot_angle_cos'] = np.cos(df['shot_angle'])  # 角度余弦
#     df['is_close'] = (df['shot_distance'] < 8).astype(int)  # 是否近距离投篮
#     df['is_very_close'] = (df['shot_distance'] < 3).astype(int)  # 是否极近距离（如扣篮）
#     df['is_end_of_period'] = (df['time_remaining'] < 10).astype(int)  # 是否节末投篮
#     df['is_critical_time'] = ((df['period'] == 4) & (df['time_remaining'] < 60)).astype(int)  # 第四节最后1分钟
#     df['year'] = pd.to_datetime(df['game_date']).dt.year  # 年份
#     df['month'] = pd.to_datetime(df['game_date']).dt.month  # 月份（反映赛季阶段）
#     df['is_home_game'] = (df['lat'] > 0).astype(int)  # 假设lat>0为主场（需根据数据实际情况调整）
#
#     # 添加交互项
#     df['distance_action'] = df['shot_distance'] * pd.factorize(df['action_type'])[0]
#     df['loc_y_action'] = df['loc_y'] * pd.factorize(df['action_type'])[0]
#     df['loc_x_action'] = df['loc_x'] * pd.factorize(df['action_type'])[0]
#     df['distance_zone'] = df['shot_distance'] * pd.factorize(df['shot_zone_basic'])[0]
#     df['time_action'] = df['time_remaining'] * pd.factorize(df['action_type'])[0]
#     df['distance_opponent'] = df['shot_distance'] * pd.factorize(df['opponent'])[0]
#     df['is_playoff_close'] = df['playoffs'] * df['is_close']
#     df['time_zone'] = df['time_remaining'] * pd.factorize(df['shot_zone_basic'])[0]  # 时间与区域交互
#     df['angle_distance'] = df['shot_angle'] * df['shot_distance']  # 角度与距离交互
#     df['loc_x_opponent'] = df['loc_x'] * pd.factorize(df['opponent'])[0]  # X位置与对手交互
#     df['period_action'] = df['period'] * pd.factorize(df['action_type'])[0]  # 节次与动作交互
#
#     # 添加三阶交互项
#     df['distance_action_zone'] = (df['shot_distance'] * pd.factorize(df['action_type'])[0] *
#                                  pd.factorize(df['shot_zone_basic'])[0])  # 距离*动作*区域
#
#     # 扩展特征列表
#     derived_features = [
#         'time_remaining', 'distance_squared', 'distance_cubed', 'distance_log',
#         'shot_angle', 'shot_angle_sin', 'shot_angle_cos', 'is_close', 'is_very_close',
#         'is_end_of_period', 'is_critical_time', 'year', 'month', 'is_home_game',
#         'distance_action', 'loc_y_action', 'loc_x_action', 'distance_zone', 'time_action',
#         'distance_opponent', 'is_playoff_close', 'time_zone', 'angle_distance',
#         'loc_x_opponent', 'period_action', 'distance_action_zone'
#     ]
#     features.extend(derived_features)
#
#     # 处理分类变量
#     categorical_cols = [
#         'action_type', 'shot_zone_basic', 'shot_zone_range', 'shot_type', 'shot_zone_area',
#         'season', 'opponent', 'playoffs', 'team_id'
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



# 借鉴别人的思路的版本---最好
import pandas as pd
import numpy as np

def prepare_data(dataset, feature_columns=None):
    """
    处理Dataset类型的数据，生成适合模型的特征矩阵X和目标向量y
    根据提供的特征工程逻辑进行修改
    """
    df = dataset.to_pandas()

    # 使用所有可能有用的原始特征（后续会筛选）
    features = [
        'minutes_remaining', 'seconds_remaining',  # 用于计算 time_remaining
        'matchup', 'season', 'game_date',  # matchup, season 和 post_achilles
        'loc_x', 'loc_y',  # 用于计算 shot_distance
        'action_type', 'shot_zone_basic', 'shot_zone_area', 'shot_type', 'opponent', 'playoffs'  # 分类特征
    ]

    # 添加衍生特征
    df['time_remaining'] = df['minutes_remaining'] * 60 + df['seconds_remaining']  # 总剩余秒数
    df['is_home_game'] = df['matchup'].apply(lambda x: 1 if 'vs.' in x else 0)  # 主场为1，客场为0
    df['season'] = df['season'].str.split('-').str[1].str[-2:]  # 提取年份后两位
    df['shot_distance'] = np.sqrt((df['loc_x'] / 10) ** 2 + (df['loc_y'] / 10) ** 2)  # 欧几里得距离
    df['game_num'] = pd.to_datetime(df['game_date']).astype('int64') // 10**9  # 转换为时间戳作为游戏编号
    df['post_achilles'] = (df['game_num'] > 1452).astype(int)  # 阿基里斯腱受伤后为1

    # 保留最终特征（移除无用特征）
    final_features = [
        'time_remaining', 'is_home_game', 'season', 'shot_distance', 'post_achilles',
        'action_type', 'shot_zone_basic', 'shot_zone_area', 'shot_type', 'opponent', 'playoffs'
    ]

    # 处理分类变量并生成哑变量
    categorical_cols = [
        'action_type', 'shot_zone_basic', 'shot_zone_area', 'shot_type', 'opponent', 'playoffs'
    ]
    X = pd.get_dummies(df[final_features], columns=categorical_cols)

    # 对齐特征
    if feature_columns is not None:
        missing_cols = set(feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[feature_columns]

    # 目标变量
    y = df['shot_made_flag'].replace('n/a', np.nan)  # 将 'n/a' 替换为 NaN
    return X, y