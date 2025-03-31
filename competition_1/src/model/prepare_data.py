import pandas as pd


def prepare_data(dataset):
    """
    处理Dataset类型的数据，将其转换为模型可用的特征矩阵X和目标向量y
    参数dataset是datasets库中的Dataset对象
    """
    # 将Dataset转换为DataFrame
    df = pd.DataFrame(dataset)

    # 选择特征
    features = ['loc_x', 'loc_y', 'shot_distance', 'minutes_remaining',
                'seconds_remaining', 'period', 'shot_type', 'shot_zone_area']

    # 处理分类变量
    df = pd.get_dummies(df[features], columns=['shot_type', 'shot_zone_area'])
    X = df  # 特征矩阵

    # 提取目标变量
    y = dataset['shot_made_flag']  # 直接从Dataset中获取目标列

    return X, y