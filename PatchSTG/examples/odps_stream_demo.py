import os
import sys
# 将上级目录添加到系统路径，以便导入 PatchSTG 模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yaml
import argparse
import logging

def setup_logger(log_path=None):
    """配置日志记录器"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=handlers
    )

def load_config(config_path):
    """加载 YAML 配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        raise

def safe_int(x, default=0):
    """安全地将值转换为整数"""
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        return int(float(x))
    except Exception:
        return default

class ODPSDynamicTurnDataset(torch.utils.data.IterableDataset):
    """
    ODPS 动态转向数据集 (流式读取版)
    继承自 IterableDataset，适用于处理无法一次性加载到内存的大规模数据集。
    """
    
    def __init__(self, odps_table, access_id=None, access_key=None, project=None, endpoint=None, limit=None, col_map=None, buffer_size=1000):
        """
        初始化 ODPS 数据集
        :param odps_table: ODPS 表名
        :param access_id: 阿里云 AccessKey ID
        :param access_key: 阿里云 AccessKey Secret
        :param project: ODPS 项目名
        :param endpoint: ODPS Endpoint
        :param limit: 限制读取的记录数 (用于测试)
        :param col_map: 列名映射配置 (JSON 字符串或字典)
        :param buffer_size: 缓冲区大小 (当前未使用，保留参数)
        """
        try:
            from odps import ODPS
        except Exception:
            raise RuntimeError('pyodps 未安装，请 pip install pyodps')
        
        # 优先使用传入的参数，否则尝试从环境变量获取
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
        self.buffer_size = buffer_size
        
        # 获取列名到索引的映射，用于后续快速访问
        self.column_names = [col.name for col in self.table.table_schema.columns]
        self.col_idx_map = {name: idx for idx, name in enumerate(self.column_names)}
        
        logging.info(f"表列名: {self.column_names}")
        logging.info(f"已初始化流式数据集，limit={self.limit}")

    def _parse_record(self, row):
        """
        解析单条 ODPS 记录为模型所需的格式
        :param row: 包含列名和值的字典
        :return: 解析后的样本字典
        """
        # 1. 解析 input_flows (交通流量数据)
        # 格式假设为: "v1,v2,...; v1,v2,..." (分号分隔不同节点/组，逗号分隔时间步)
        input_flows = row.get('input_flows', '')
        groups = [g.strip() for g in str(input_flows).split(';') if g.strip()]
        
        node_arrays = []
        for g in groups:
            vals = [v.strip() for v in g.split(',')]
            # 确保每个节点有 13 个时间步的数据 (前12个为历史，第13个为标签)
            if len(vals) < 13:
                vals = vals + ['0'] * (13 - len(vals))
            else:
                vals = vals[:13]
            nums = [float(x) if x else 0.0 for x in vals]
            node_arrays.append(nums)
        
        if not node_arrays:
            node_arrays = [[0.0] * 13]
        
        node_arrays = np.array(node_arrays, dtype=np.float32)
        
        # 2. 解析 time_features (时间特征)
        # 格式假设与 input_flows 类似
        time_feats = row.get('time_features', '')
        segs = [s.strip() for s in str(time_feats).split(';') if s.strip()]
        
        time_feats_arr = []
        for seg in segs:
            parts = seg.split()
            # 每个时间步应有 6 个特征 (weekday, hour, minute, day_type, day, month)
            if len(parts) >= 6:
                nums = [float(x) for x in parts[:6]]
            else:
                nums = [float(x) for x in parts] + [0.0] * (6 - len(parts))
            time_feats_arr.append(nums)
        
        # 补齐到 13 个时间步
        while len(time_feats_arr) < 13:
            time_feats_arr.append([0.0] * 6)
        
        time_feats_arr = np.array(time_feats_arr[:13], dtype=np.float32)
        
        # 3. 提取标签 (Label)
        # 假设取第一个节点的第 13 个时间步的值作为预测目标
        label = float(node_arrays[0, 12]) if node_arrays.shape[0] > 0 else 0.0
        
        # 4. 解析其他元数据
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

    def __iter__(self):
        """
        迭代器方法，实现流式读取
        """
        worker_info = torch.utils.data.get_worker_info()
        
        # 简单的单进程/多进程切分策略
        # 注意：ODPS open_reader 可能不支持高效的随机 seek，这里假设 num_workers=0 或 1
        # 如果必须多 worker，建议在 ODPS 侧预先分片或使用 tunnel 的分片读取功能
        
        if worker_info is not None and worker_info.num_workers > 1:
            logging.warning("ODPS 流式读取在多 worker 模式下可能需要更复杂的切分策略，当前仅建议 num_workers=0")
        
        count = 0
        try:
            # 打开 ODPS 表读取器
            with self.table.open_reader() as reader:
                for record in reader:
                    if self.limit and count >= self.limit:
                        break
                    
                    try:
                        # 将 Record 对象转换为字典
                        row = {}
                        for col_name in self.column_names:
                            col_idx = self.col_idx_map[col_name]
                            row[col_name] = record[col_idx]
                        
                        # 应用列名映射（如果有）
                        if self.col_map:
                            import json
                            mapd = json.loads(self.col_map) if isinstance(self.col_map, str) else self.col_map
                            row = {mapd.get(k, k): v for k, v in row.items()}
                        
                        # 解析并 yield 数据
                        yield self._parse_record(row)
                        count += 1
                        
                    except Exception as e:
                        logging.error(f"解析记录失败: {e}")
                        continue
                        
        except Exception as e:
            logging.error(f"ODPS 读取流中断: {e}")
            raise

def dynamic_turn_collate_fn(batch):
    """
    自定义 Collate 函数，用于将一个批次的样本整理成模型输入的 Tensor
    处理变长的节点数量 (通过 padding)
    """
    # 找出该批次中最大的节点数
    max_nodes = int(max(item['node_count'] for item in batch))
    batch_size = len(batch)

    # 初始化 input_flows 容器: [B, max_nodes, 13]
    input_flows = np.zeros((batch_size, max_nodes, 13), dtype=np.float32)
    # 初始化掩码 mask: [B, max_nodes] (1 表示真实节点，0 表示 padding)
    mask = np.zeros((batch_size, max_nodes), dtype=np.float32)
    
    for i, item in enumerate(batch):
        n = int(item['node_count'])
        actual_n = min(n, item['input_flows'].shape[0])
        # 填充数据
        input_flows[i, :actual_n, :] = item['input_flows'][:actual_n, :]
        # 设置掩码
        mask[i, :actual_n] = 1.0

    # 构建模型输入 x: 取前 12 个时间步作为历史输入
    hist_steps = 12
    x_np = input_flows[:, :, :hist_steps]  # [B, N, 12]
    x_np = np.transpose(x_np, (0, 2, 1))   # [B, 12, N]
    x_np = x_np[..., np.newaxis]           # [B, 12, N, 1]

    # 构建时间特征 te: [B, 13, 6] -> 取前 12 步并广播到所有节点 -> [B, 12, N, 6]
    time_features = np.stack([item['time_features'] for item in batch], axis=0)  # [B, 13, 6]
    te_np = time_features[:, :hist_steps, :]  # [B, 12, 6]
    te_np = np.expand_dims(te_np, 2)          # [B, 12, 1, 6]
    te_np = np.repeat(te_np, max_nodes, axis=2)  # [B, 12, N, 6]

    # 构建标签 labels: 取第 13 个时间步 (索引 12) -> [B, N, 1]
    label_np = input_flows[:, :, hist_steps]  # [B, N]
    label_np = label_np[..., np.newaxis].astype(np.float32)  # [B, N, 1]

    adcodes = np.array([item['adcode'] for item in batch], dtype=np.int64)

    # 转换为 PyTorch Tensor
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
    parser.add_argument('--device', type=str, default='cpu', help='运行设备')
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

    # 如果没有真实 dataset，则构造一个小的合成数据集 (SyntheticDataset)
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
    logging.info(f'模型预测输出 shape: {pred.shape}')

    # 取最后一步预测并计算 mse
    pred_last = pred[:, -1, :, :]
    # pred_last: [B, N, out_dim], labels_t: [B, N, 1]
    try:
        pred_val = pred_last[..., 0]
        label_val = labels_t[..., 0]
        mse = torch.nn.functional.mse_loss(pred_val, label_val)
        logging.info(f'预测值与标签的简单 MSE 损失: {mse.item():.6f}')
    except Exception as e:
        logging.warning(f'计算 loss 失败: {e}')

    logging.info('演示完成（前向与 shape 检查）')

if __name__ == '__main__':
    main()
