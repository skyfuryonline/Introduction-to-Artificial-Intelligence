from datasets import load_dataset

train_data = load_dataset("json",data_files="../data/train.jsonl",split='train')
val_data = load_dataset("json",data_files="../data/val.jsonl",split='train')
test_data = load_dataset("json",data_files="../data/test.jsonl",split='train')
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
