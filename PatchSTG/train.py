import os
import sys
import time
import argparse
import logging
import yaml

# 将项目根目录添加到系统路径，以便导入模块
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 尝试导入 ODPS 数据集类
try:
    from examples.odps_stream_demo import ODPSDynamicTurnDataset, dynamic_turn_collate_fn
except Exception:
    # 如果 examples 不是包路径，尝试直接导入
    from odps_stream_demo import ODPSDynamicTurnDataset, dynamic_turn_collate_fn

# 导入 PatchSTG 模型
from models.model import PatchSTG


def setup_logger(log_path=None):
    """
    配置日志记录器
    :param log_path: 日志文件路径或目录
    """
    # 生成带时间戳的日志文件名
    ts = time.strftime('%Y%m%d_%H%M%S')
    if log_path:
        # 如果提供了路径，检查是目录还是文件
        log_dir = log_path if os.path.isdir(log_path) else os.path.dirname(log_path) or '.'
    else:
        # 默认在当前脚本目录下的 logs 文件夹
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{ts}.log')

    # 移除现有的处理器，避免重复日志
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    # 设置日志格式和处理器（同时输出到控制台和文件）
    fmt = '%(asctime)s %(levelname)s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)
    logging.info(f'日志将输出到标准输出和文件: {log_file}')


def masked_mae(pred, target, mask, eps=1e-6):
    """
    计算掩码平均绝对误差 (Masked MAE)
    :param pred: 预测值 [B, N]
    :param target: 真实值 [B, N]
    :param mask: 掩码 [B, N] (1 表示有效数据，0 表示缺失/填充)
    :param eps: 防止除零的微小值
    :return: 标量损失值
    """
    diff = torch.abs(pred - target) * mask
    denom = mask.sum() + eps
    return diff.sum() / denom


def load_config(path):
    """加载 YAML 配置文件"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/train.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    # 加载配置
    cfg = {}
    if os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        print(f"配置文件 {args.config} 未找到，使用默认设置")

    # 设置日志
    log_path = cfg.get('log', None)
    setup_logger(log_path)
    device = torch.device(args.device)

    # 获取训练参数
    batch_size = cfg.get('batch_size', 32)
    epochs = cfg.get('epochs', 3)
    lr = cfg.get('lr', 1e-3)
    limit = cfg.get('limit', None)

    # 构建数据集
    try:
        dataset = ODPSDynamicTurnDataset(
            odps_table=cfg.get('odps_table'),
            access_id=cfg.get('access_id'),
            access_key=cfg.get('access_key'),
            project=cfg.get('project'),
            endpoint=cfg.get('endpoint'),
            limit=limit,
            col_map=cfg.get('col_map', None)
        )
        logging.info('成功从 ODPS 构建数据集')
    except Exception as e:
        logging.warning(f'构建 ODPS 数据集失败: {e}')
        logging.info('回退到小型合成数据集用于调试')

        # 定义一个用于调试的伪数据集
        class _Dummy(Dataset):
            def __len__(self):
                return 16

            def __getitem__(self, idx):
                # 生成 3 个节点，13 个时间步的数据
                input_flows = [[0.0]*13 for _ in range(3)]
                time_features = [[0,0,0,0,0,0] for _ in range(13)]
                return {'input_flows': input_flows, 'time_features': time_features, 'label': 0.0, 'node_count': 3, 'adcode': 0}

        from torch.utils.data import Dataset as _Dataset
        dataset = _Dummy()

    # 创建数据加载器
    # 注意：IterableDataset (流式数据集) 不支持 shuffle=True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dynamic_turn_collate_fn)

    # 查看一个批次的数据以推断节点数量和形状
    try:
        batch = next(iter(dataloader))
        x, te, mask, labels, adcodes = batch
        # x: [B, T, N, 1], te: [B, T, N, 6], labels: [B, N, 1]
        B, T, N, C = x.shape
        logging.info(f'批次数据形状预览: x={x.shape}, te={te.shape}, labels={labels.shape}, mask={mask.shape}')
    except Exception as e:
        logging.error(f'从数据加载器获取批次失败: {e}')
        raise

    # 使用推断出的节点数 N 初始化模型
    output_len = cfg.get('output_len', 1)
    layers = cfg.get('layers', 2)
    input_dims = cfg.get('input_dims', 8)
    node_dims = cfg.get('node_dims', 16)
    time_dims = cfg.get('time_dims', 4)

    model = PatchSTG(output_len=output_len, node_num=N, layers=layers, input_dims=input_dims, node_dims=node_dims, time_dims=time_dims)
    model.to(device)
    logging.info('模型初始化完成')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early Stopping 参数
    patience = cfg.get('early_stopping_patience', 10)  # 容忍多少个 epoch 没有改善
    min_delta = cfg.get('early_stopping_min_delta', 1e-4)  # 最小改善阈值
    best_loss = float('inf')
    patience_counter = 0
    
    # 训练循环
    model.train()
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for i, batch in enumerate(dataloader):
            x, te, mask, labels, adcodes = batch
            x = x.to(device)
            te = te.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            pred = model(x, te, mask=mask)  # 输出形状: [B, T, N, out_dim]
            
            # 取最后一个时间步和第一个输出维度作为预测结果
            pred_last = pred[:, -1, :, 0]  # [B, N]
            target = labels[..., 0]       # [B, N]
            
            loss = masked_mae(pred_last, target, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (i+1) % 10 == 0:
                logging.info(f'Epoch {epoch} 批次 {i+1} loss={loss.item():.6f}')

        dt = time.time() - t0
        avg_loss = epoch_loss / max(1, n_batches)
        logging.info(f'Epoch {epoch} 完成: 平均 loss={avg_loss:.6f} 耗时={dt:.1f}s')

        # Early Stopping 检查
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            logging.info(f'Loss 改善至 {best_loss:.6f}，重置 patience 计数器')
        else:
            patience_counter += 1
            logging.info(f'Loss 未改善 ({patience_counter}/{patience})')
            
            if patience_counter >= patience:
                logging.info(f'Early Stopping: {patience} 个 epoch 内 loss 未改善，停止训练')
                break

        # 保存检查点 (确保父目录存在)
        ckpt = {'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        # 支持配置文件中的 {epoch} 占位符
        ckpt_path_template = cfg.get('ckpt_path', f'checkpoint_epoch{epoch}.pth')
        ckpt_path = ckpt_path_template.replace('{epoch}', str(epoch))
        try:
            ckpt_dir = os.path.dirname(ckpt_path) or '.'
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(ckpt, ckpt_path)
            logging.info(f'检查点已保存至 {ckpt_path}')
        except Exception:
            logging.exception(f'保存检查点至 {ckpt_path} 失败; 尝试保存到当前目录')
            fallback = f'checkpoint_epoch{epoch}_{int(time.time())}.pth'
            try:
                torch.save(ckpt, fallback)
                logging.info(f'检查点已保存至备用路径 {fallback}')
            except Exception:
                logging.exception(f'保存备用检查点 {fallback} 失败')

    logging.info('训练结束')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # 确保任何未捕获的异常也被记录到日志文件
        logging.exception('train.py 中发生未捕获异常')
        raise
