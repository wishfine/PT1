import torch
from lib.odps_dataset import dynamic_turn_collate_fn

# 用你给的示例数据，模拟 DataLoader 取到的原始 batch
raw_batch = [
    {
        'sample_id': '310000_5130138137130960258_5130138134983476751_2025-10-10_175100',
        'adcode': 310000,
        'start_time': '2025-10-10 17:51:00',
        'input_flows': '0,1,0,0,3,0,0,4,0,0,15,2,1;0,0,0,0,0,0,0,0,0,0,0,0,0',
        'time_features': '5 17 39 0 9 9;5 17 40 0 9 9;5 17 41 0 9 9;5 17 42 0 9 9;5 17 43 0 9 9;5 17 44 0 9 9;5 17 45 0 9 9;5 17 46 0 9 9;5 17 47 0 9 9;5 17 48 0 9 9;5 17 49 0 9 9;5 17 50 0 9 9;5 17 51 0 9 9',
        'node_pairs': '(5130138137130960258,5130138134983476751)',
        'node_count': 2,
        'sample_date': '20251010',
        'sample_time_of_day': '17:51',
    },
    {
        'sample_id': '310000_5130138139278443520_5130138139278442622_2025-08-06_171500',
        'adcode': 310000,
        'start_time': '2025-08-06 17:15:00',
        'input_flows': '0,0,0,0,0,0,0,0,0,0,0,0,1;0,0,0,0,0,0,0,0,0,0,0,0,0',
        'time_features': '3 17 3 0 5 7;3 17 4 0 5 7;3 17 5 0 5 7;3 17 6 0 5 7;3 17 7 0 5 7;3 17 8 0 5 7;3 17 9 0 5 7;3 17 10 0 5 7;3 17 11 0 5 7;3 17 12 0 5 7;3 17 13 0 5 7;3 17 14 0 5 7;3 17 15 0 5 7',
        'node_pairs': '(5130138139278443520,5130138139278442622)',
        'node_count': 2,
        'sample_date': '20250806',
        'sample_time_of_day': '17:15',
    },
]

# 适配 dynamic_turn_collate_fn 的输入格式
# 假设 collate_fn 支持 dict list 输入，否则需适配

def main():
    batch = dynamic_turn_collate_fn(raw_batch)
    batch_flow, batch_static_feat, batch_time_features, batch_city_id, batch_label, batch_mask = batch
    print('batch_flow:', batch_flow.shape)
    print('batch_static_feat:', batch_static_feat.shape)
    print('batch_time_features:', batch_time_features.shape)
    print('batch_city_id:', batch_city_id.shape)
    print('batch_label:', batch_label.shape)
    print('batch_mask:', batch_mask.shape if batch_mask is not None else None)
    # 打印部分内容
    print('batch_flow[0]:', batch_flow[0])
    print('batch_time_features[0]:', batch_time_features[0])
    print('batch_mask[0]:', batch_mask[0] if batch_mask is not None else None)

if __name__ == '__main__':
    main()
