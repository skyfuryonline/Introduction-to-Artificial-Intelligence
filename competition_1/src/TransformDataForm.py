# 将数据从csv转换为jsonl


from datasets import load_dataset

data = load_dataset("csv",data_files="../data/*.csv",split='train')
'''
Dataset({
    features: ['action_type', 'combined_shot_type', 'game_event_id', 'game_id', 'lat', 'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'period', 'playoffs', 'season', 'seconds_remaining', 'shot_distance', 'shot_made_flag', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id'],
    num_rows: 30697
})
'''
# 按 8:2 的比例划分
train_test = data.train_test_split(train_size=0.8)


# 获取 train 和 val 数据
train_data = train_test["train"]
val_data = train_test["test"]

# print(train_data)
# print(val_data)
'''
Dataset({
    features: ['action_type', 'combined_shot_type', 'game_event_id', 'game_id', 'lat', 'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'period', 'playoffs', 'season', 'seconds_remaining', 'shot_distance', 'shot_made_flag', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id'],
    num_rows: 24557
})
Dataset({
    features: ['action_type', 'combined_shot_type', 'game_event_id', 'game_id', 'lat', 'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'period', 'playoffs', 'season', 'seconds_remaining', 'shot_distance', 'shot_made_flag', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id'],
    num_rows: 6140
})
'''

output_train_jsonl = '../data/train.jsonl'
output_val_jsonl = '../data/val.jsonl'

import json
# 打开输出文件
with open(output_train_jsonl, 'w', encoding='utf-8') as jsonl_file:
    # 遍历 Dataset 中的每一行
    for row in train_data:
        # row 已经是字典格式，直接转换为 JSON 字符串
        jsonl_string = json.dumps(row)
        # 写入文件，每行一个 JSON 对象
        jsonl_file.write(jsonl_string + '\n')

print(f"转换完成！已将 Dataset 数据转换为 {output_train_jsonl}")
# 打开输出文件
with open(output_val_jsonl, 'w', encoding='utf-8') as jsonl_file:
    # 遍历 Dataset 中的每一行
    for row in val_data:
        # row 已经是字典格式，直接转换为 JSON 字符串
        jsonl_string = json.dumps(row)
        # 写入文件，每行一个 JSON 对象
        jsonl_file.write(jsonl_string + '\n')

print(f"转换完成！已将 Dataset 数据转换为 {output_val_jsonl}")
