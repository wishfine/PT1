import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yaml
import argparse
import logging

def setup_logger(log_path=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=handlers
    )

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        raise

def safe_int(x, default=0):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        return int(float(x))
    except Exception:
        return default

class ODPSDynamicTurnDataset(Dataset):
    """ODPS 动态转向数据集"""
    
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
        
        # 获取列名到索引的映射
        self.column_names = [col.name for col in self.table.table_schema.columns]
        self.col_idx_map = {name: idx for idx, name in enumerate(self.column_names)}
        
        logging.info(f"表列名: {self.column_names}")
        
        self._load_rows()

    def _load_rows(self):
        """加载数据行"""
        self._rows = []
        
        logging.info(f"开始加载数据，limit={self.limit}...")
        
        try:
            with self.table.open_reader() as reader:
                for i, record in enumerate(reader):
                    if self.limit and i >= self.limit:
                        break
                    
                    try:
                        # 使用索引访问 Record（不是 rec.get()）
                        row = {}
                        for col_name in self.column_names:
                            col_idx = self.col_idx_map[col_name]
                            row[col_name] = record[col_idx]
                        
                        # 应用列名映射（如果有）
                        if self.col_map:
                            import json
                            mapd = json.loads(self.col_map) if isinstance(self.col_map, str) else self.col_map
                            row = {mapd.get(k, k): v for k, v in row.items()}
                        
                        self._rows.append(row)
                        
                        # 每1000条打印进度
                        if (i + 1) % 1000 == 0:
                            logging.info(f"  已加载 {i + 1} 条...")
                            
                    except Exception as e:
                        logging.error(f"处理第 {i} 条记录时出错: {e}")
                        logging.error(f"Record 类型: {type(record)}")
                        logging.error(f"Record 可用方法: {[m for m in dir(record) if not m.startswith('_')]}")
                        raise
            
            logging.info(f"数据加载完成，共 {len(self._rows)} 条")
            
        except Exception as e:
            logging.error(f"_load_rows 失败: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise

    def __len__(self):
        return len(self._rows) if self._rows else 0

    def __getitem__(self, idx):
        row = self._rows[idx]
        
        # 解析 input_flows
        input_flows = row.get('input_flows', '')
        groups = [g.strip() for g in str(input_flows).split(';') if g.strip()]
        
        node_arrays = []
        for g in groups:
            vals = [v.strip() for v in g.split(',')]
            if len(vals) < 13:
                vals = vals + ['0'] * (13 - len(vals))
            else:
                vals = vals[:13]
            nums = [float(x) if x else 0.0 for x in vals]
            node_arrays.append(nums)
        
        if not node_arrays:
            node_arrays = [[0.0] * 13]
        
        node_arrays = np.array(node_arrays, dtype=np.float32)
        
        # 解析时间特征
        time_feats = row.get('time_features', '')
        segs = [s.strip() for s in str(time_feats).split(';') if s.strip()]
        
        time_feats_arr = []
        for seg in segs:
            parts = seg.split()
            if len(parts) >= 6:
                nums = [float(x) for x in parts[:6]]
            else:
                nums = [float(x) for x in parts] + [0.0] * (6 - len(parts))
            time_feats_arr.append(nums)
        
        # 补齐到13个时间步
        while len(time_feats_arr) < 13:
            time_feats_arr.append([0.0] * 6)
        
        time_feats_arr = np.array(time_feats_arr[:13], dtype=np.float32)
        
        # label: 第一组的第13个值
        label = float(node_arrays[0, 12]) if node_arrays.shape[0] > 0 else 0.0
        
        # 其他字段
        node_count = safe_int(row.get('node_count'), node_arrays.shape[0])
        sample_id = str(row.get('sample_id', ''))
        adcode = safe_int(row.get('adcode'), 0)
        sample_date = str(row.get('sample_date', ''))
        
        return {
            'input_flows': node_arrays,
            'time_features': time_feats_arr,
            'label': label,
            'node_count': node_count,
            'adcode': adcode,
            'sample_id': sample_id,
            'sample_date': sample_date
        }

def dynamic_turn_collate_fn(batch):
    """自定义 collate 函数"""
    # We treat the first 12 minutes as input (history) and the 13th minute as label
    max_nodes = int(max(item['node_count'] for item in batch))
    batch_size = len(batch)

    # input_flows: (B, max_nodes, 13)
    input_flows = np.zeros((batch_size, max_nodes, 13), dtype=np.float32)
    mask = np.zeros((batch_size, max_nodes), dtype=np.float32)
    for i, item in enumerate(batch):
        n = int(item['node_count'])
        actual_n = min(n, item['input_flows'].shape[0])
        input_flows[i, :actual_n, :] = item['input_flows'][:actual_n, :]
        mask[i, :actual_n] = 1.0

    # Build x: history 12 steps -> [B, 12, N, 1]
    hist_steps = 12
    x_np = input_flows[:, :, :hist_steps]  # [B, N, 12]
    x_np = np.transpose(x_np, (0, 2, 1))   # [B, 12, N]
    x_np = x_np[..., np.newaxis]           # [B, 12, N, 1]

    # time_features: (B, 13, 6) -> take first 12 rows and broadcast to nodes -> [B,12,N,6]
    time_features = np.stack([item['time_features'] for item in batch], axis=0)  # [B, 13, 6]
    te_np = time_features[:, :hist_steps, :]  # [B,12,6]
    te_np = np.expand_dims(te_np, 2)  # [B,12,1,6]
    te_np = np.repeat(te_np, max_nodes, axis=2)  # [B,12,N,6]

    # labels: take the 13th minute (index 12) per node -> [B, N, 1]
    label_np = input_flows[:, :, hist_steps]  # [B, N]
    label_np = label_np[..., np.newaxis].astype(np.float32)  # [B, N, 1]

    adcodes = np.array([item['adcode'] for item in batch], dtype=np.int64)

    # convert to tensors
    x = torch.from_numpy(x_np).float()
    te = torch.from_numpy(te_np).long()
    mask = torch.from_numpy(mask).float()
    labels = torch.from_numpy(label_np).float()
    adcodes = torch.from_numpy(adcodes)

    return x, te, mask, labels, adcodes

def main():
    parser = argparse.ArgumentParser(description='ODPS 流式 PatchSTG demo')
    parser.add_argument('--config', type=str, default=None, help='YAML 配置文件路径')
    parser.add_argument('--log', type=str, default=None, help='日志文件路径')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    args = parser.parse_args()

    # 日志初始化
    setup_logger(args.log)

    # 配置文件路径（可选）
    config_path = args.config or os.path.join(os.path.dirname(__file__), 'odps_stream_config.yaml')
    try:
        config = load_config(config_path)
    except Exception:
        logging.warning('未找到或无法加载配置，使用默认演示参数')
        config = {}

    # 尝试使用 ODPS 数据集；若失败，则使用伪数据回退（便于本地调试）
    dataset = None
    try:
        dataset = ODPSDynamicTurnDataset(
            odps_table=config.get('odps_table', ''),
            access_id=config.get('access_id'),
            access_key=config.get('access_key'),
            project=config.get('project'),
            endpoint=config.get('endpoint'),
            limit=config.get('limit', 4)
        )
        logging.info('✓ ODPS 数据集构建成功')
    except Exception as e:
        logging.warning(f'无法使用 ODPS 数据集（{e}），回退到伪数据用于演示')

    # 如果没有真实 dataset，则构造一个小的 synthetic dataset
    if dataset is None:
        class SyntheticDataset(Dataset):
            def __init__(self, samples=8, max_nodes=4):
                self.samples = samples
                self.max_nodes = max_nodes
            def __len__(self):
                return self.samples
            def __getitem__(self, idx):
                # 每个样本 N 随机 1..max_nodes
                n = (idx % self.max_nodes) + 1
                flows = np.random.rand(n, 13).astype(np.float32) * 100
                # time features: 13 x 6
                te = np.zeros((13,6), dtype=np.float32)
                for i in range(13):
                    te[i] = np.array([i%7, i%24, i%60, idx%9, (i%31), (i%12)], dtype=np.float32)
                return {
                    'input_flows': flows,
                    'time_features': te,
                    'label': float(flows[0,12]) if flows.shape[0]>0 else 0.0,
                    'node_count': n,
                    'adcode': 0,
                    'sample_id': f'synth_{idx}',
                    'sample_date': '1970-01-01'
                }

        dataset = SyntheticDataset(samples=config.get('limit', 8), max_nodes=config.get('max_nodes', 4))
        logging.info('✓ 使用合成数据集进行演示')

    # 小批量 DataLoader（用于快速验证）
    batch_size = config.get('batch_size', 2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=dynamic_turn_collate_fn)

    # 读取第一个 batch 并构建模型
    try:
        batch = next(iter(dataloader))
    except Exception as e:
        logging.error(f'无法从 DataLoader 获取批次: {e}')
        return

    # collate 返回: x, te, mask, labels, adcodes
    x, te, mask, labels, adcodes = batch
    logging.info(f'批次 shapes: x={x.shape}, te={te.shape}, labels={labels.shape}, mask={mask.shape}, adcodes={adcodes.shape}')

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    # 基于 batch 中节点数构造模型（node_num 取批次中的 N）
    _, T, N, C = x.shape
    node_num = N
    # 模型参数（可从 config 覆盖）
    model_cfg = {
        'output_len': config.get('output_len', 1),
        'node_num': node_num,
        'layers': config.get('model_layers', 2),
        'input_dims': config.get('input_dims', 16),
        'node_dims': config.get('node_dims', 16),
        'time_dims': config.get('time_dims', 4),
    }

    # 延迟导入模型以避免循环依赖（models 在项目顶层）
    try:
        from PatchSTG.models.model import PatchSTG
    except Exception:
        # 如果包路径不同，尝试相对导入
        try:
            from models.model import PatchSTG
        except Exception as e:
            logging.error(f'无法导入 PatchSTG 模型: {e}')
            return

    model = PatchSTG(model_cfg['output_len'], model_cfg['node_num'], model_cfg['layers'],
                     model_cfg['input_dims'], model_cfg['node_dims'], model_cfg['time_dims'])
    model.to(device)

    # 前向一次并计算简单 loss（以最后一时间步的预测与 labels 对齐）
    x_t = x.to(device)
    te_t = te.to(device)
    mask_t = mask.to(device)
    labels_t = labels.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(x_t, te_t, mask_t)  # [B, T, N, out_dim]
    logging.info(f'model pred shape: {pred.shape}')

    # 取最后一步预测并计算 mse
    pred_last = pred[:, -1, :, :]
    # pred_last: [B, N, out_dim], labels_t: [B, N, 1]
    try:
        pred_val = pred_last[..., 0]
        label_val = labels_t[..., 0]
        mse = torch.nn.functional.mse_loss(pred_val, label_val)
        logging.info(f'Simple MSE between pred_last and label: {mse.item():.6f}')
    except Exception as e:
        logging.warning(f'计算 loss 失败: {e}')

    logging.info('演示完成（前向与 shape 检查）')

if __name__ == '__main__':
    main()

