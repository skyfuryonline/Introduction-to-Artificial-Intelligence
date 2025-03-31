from datasets import load_dataset

train_data = load_dataset("json",data_files=r"D:\PycharmProjects\Introduction-to-Artificial-Intelligence\competition_1\data\train.jsonl",split='train')
val_data = load_dataset("json",data_files=r"D:\PycharmProjects\Introduction-to-Artificial-Intelligence\competition_1\data\val.jsonl",split='train')
test_data = load_dataset("json",data_files=r"D:\PycharmProjects\Introduction-to-Artificial-Intelligence\competition_1\data\test.jsonl",split='train')
# print(train_data)
# print(val_data)
# print(test_data)
'''
Dataset({
    features: ['action_type', 'combined_shot_type', 'game_event_id', 'game_id', 'lat', 'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'period', 'playoffs', 'season', 'seconds_remaining', 'shot_distance', 'shot_made_flag', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id', '__index_level_0__'],
    num_rows: 20557
})
Dataset({
    features: ['action_type', 'combined_shot_type', 'game_event_id', 'game_id', 'lat', 'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'period', 'playoffs', 'season', 'seconds_remaining', 'shot_distance', 'shot_made_flag', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id', '__index_level_0__'],
    num_rows: 5140
})
Dataset({
    features: ['action_type', 'combined_shot_type', 'game_event_id', 'game_id', 'lat', 'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'period', 'playoffs', 'season', 'seconds_remaining', 'shot_distance', 'shot_made_flag', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id', '__index_level_0__'],
    num_rows: 5000
})
'''
