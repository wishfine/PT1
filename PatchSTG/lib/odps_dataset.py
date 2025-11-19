import os
import numpy as np
import torch
from torch.utils.data import Dataset

def safe_int(x, default=0):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        return int(float(x))
    except Exception:
        return default

class ODPSDynamicTurnDataset(Dataset):
    """
    支持 ODPS 流式读取的 PyTorch Dataset，适配 variable-length input_flows（node_count 不同）。
    需配合自定义 collate_fn 使用。
    """
    def __init__(self, odps_table, access_id=None, access_key=None, project=None, endpoint=None, limit=None, col_map=None):
        try:
            from odps import ODPS
        except Exception:
            raise RuntimeError('pyodps 未安装，请 pip install pyodps')
        access_id = access_id or os.environ.get('ODPS_ACCESS_ID')
        access_key = access_key or os.environ.get('ODPS_ACCESS_KEY')
        project = project or os.environ.get('ODPS_PROJECT')
        endpoint = endpoint or os.environ.get('ODPS_ENDPOINT')
        if not (access_id and access_key and project and endpoint):
            raise RuntimeError('ODPS 连接信息缺失')
        self.odps = ODPS(access_id, access_key, project=project, endpoint=endpoint)
        self.table = self.odps.get_table(odps_table)
        self.limit = limit
        self.col_map = col_map
        self._rows = None
        self._load_rows()

    def _load_rows(self):
        # 预加载所有行（可按需优化为流式）
        self._rows = []
        with self.table.open_reader() as reader:
            for i, rec in enumerate(reader):
                if self.limit and i >= self.limit:
                    break
                row = {col.name: rec.get(col.name) for col in reader.schema.columns}
                if self.col_map:
                    import json
                    mapd = json.loads(self.col_map) if isinstance(self.col_map, str) else self.col_map
                    row = {mapd.get(k, k): v for k, v in row.items()}
                self._rows.append(row)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        row = self._rows[idx]
        # 解析 input_flows
        input_flows = row['input_flows']
        groups = [g.strip() for g in str(input_flows).split(';') if g.strip() != '']
        node_arrays = []
        for g in groups:
            vals = [v.strip() for v in g.split(',')]
            if len(vals) < 13:
                vals = vals + ['0'] * (13 - len(vals))
            else:
                vals = vals[:13]
            nums = [float(x) if x != '' else 0.0 for x in vals]
            node_arrays.append(nums)
        node_arrays = np.array(node_arrays, dtype=np.float32)  # shape: (node_count, 13)
        # 解析时间特征
        time_feats = row.get('time_features', '')
        segs = [s.strip() for s in str(time_feats).split(';') if s.strip() != '']
        time_feats_arr = []
        for seg in segs:
            parts = [float(x) for x in seg.split()[:6]]
            time_feats_arr.append(parts)
        time_feats_arr = np.array(time_feats_arr, dtype=np.float32)  # shape: (13, 6)
        # label: 当前转向的 t0 流量（第1组的第13个值）
        label = node_arrays[0, 12]
        # 其它字段
        node_count = safe_int(row.get('node_count', node_arrays.shape[0]), node_arrays.shape[0])
        sample_id = row.get('sample_id', None)
        adcode = safe_int(row.get('adcode', 0), 0)
        sample_date = row.get('sample_date', '')
        return {
            'input_flows': node_arrays,  # (node_count, 13)
            'time_features': time_feats_arr,  # (13, 6)
            'label': label,
            'node_count': node_count,
            'adcode': adcode,
            'sample_id': sample_id,
            'sample_date': sample_date
        }

def dynamic_turn_collate_fn(batch):
    """
    batch: list of dicts from ODPSDynamicTurnDataset
    自动对 input_flows 做 padding，生成 mask
    """
    max_nodes = max(item['node_count'] for item in batch)
    batch_size = len(batch)
    # input_flows: (B, max_nodes, 13)
    input_flows = np.zeros((batch_size, max_nodes, 13), dtype=np.float32)
    mask = np.zeros((batch_size, max_nodes), dtype=np.float32)
    for i, item in enumerate(batch):
        n = item['node_count']
        input_flows[i, :n, :] = item['input_flows']
        mask[i, :n] = 1.0
    # time_features: (B, 13, 6)
    time_features = np.stack([item['time_features'] for item in batch], axis=0)
    labels = np.array([item['label'] for item in batch], dtype=np.float32)
    adcodes = np.array([item['adcode'] for item in batch], dtype=np.int64)
    return {
        'input_flows': torch.from_numpy(input_flows),  # (B, max_nodes, 13)
        'mask': torch.from_numpy(mask),                # (B, max_nodes)
        'time_features': torch.from_numpy(time_features),  # (B, 13, 6)
        'labels': torch.from_numpy(labels),            # (B,)
        'adcodes': torch.from_numpy(adcodes),          # (B,)
    }
