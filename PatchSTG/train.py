import os
import sys
import time
import argparse
import logging
import yaml

# make project root importable
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# import dataset and collate from examples
try:
    from examples.odps_stream_demo import ODPSDynamicTurnDataset, dynamic_turn_collate_fn
except Exception:
    # fallback if examples is not a package path
    from odps_stream_demo import ODPSDynamicTurnDataset, dynamic_turn_collate_fn

# import model
from models.model import PatchSTG


def setup_logger(log_path=None):
    # Ensure a timestamped train log filename and that the directory exists.
    # If log_path is a directory or a file path, create a file named train_YYYYmmdd_HHMMSS.log inside it
    ts = time.strftime('%Y%m%d_%H%M%S')
    if log_path:
        # if user provided a directory, use it; if a file, use its directory
        log_dir = log_path if os.path.isdir(log_path) else os.path.dirname(log_path) or '.'
    else:
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{ts}.log')

    # Remove existing handlers to avoid duplicate logs when called multiple times
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    fmt = '%(asctime)s %(levelname)s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)
    logging.info(f'Logging to stdout and file: {log_file}')


def masked_mae(pred, target, mask, eps=1e-6):
    # pred: [B, N], target: [B, N], mask: [B, N]
    diff = torch.abs(pred - target) * mask
    denom = mask.sum() + eps
    return diff.sum() / denom


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/train.yaml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # load config
    cfg = {}
    if os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        print(f"配置文件 {args.config} 未找到，使用默认设置")

    log_path = cfg.get('log', None)
    setup_logger(log_path)
    device = torch.device(args.device)

    batch_size = cfg.get('batch_size', 32)
    epochs = cfg.get('epochs', 3)
    lr = cfg.get('lr', 1e-3)
    limit = cfg.get('limit', None)

    # build dataset
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
        logging.info('Dataset constructed from ODPS')
    except Exception as e:
        logging.warning(f'Failed to build ODPS dataset: {e}')
        logging.info('Falling back to small synthetic dataset for debug')

        class _Dummy(Dataset):
            def __len__(self):
                return 16

            def __getitem__(self, idx):
                # produce 3 nodes, 13 timesteps
                input_flows = [[0.0]*13 for _ in range(3)]
                time_features = [[0,0,0,0,0,0] for _ in range(13)]
                return {'input_flows': input_flows, 'time_features': time_features, 'label': 0.0, 'node_count': 3, 'adcode': 0}

        from torch.utils.data import Dataset as _Dataset
        dataset = _Dummy()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=dynamic_turn_collate_fn)

    # peek one batch to infer node_num and shapes
    try:
        batch = next(iter(dataloader))
        x, te, mask, labels, adcodes = batch
        # x: [B, T, N, 1], te: [B, T, N, 6], labels: [B, N, 1]
        B, T, N, C = x.shape
        logging.info(f'Peek batch shapes: x={x.shape}, te={te.shape}, labels={labels.shape}, mask={mask.shape}')
    except Exception as e:
        logging.error(f'Failed to fetch batch from dataloader: {e}')
        raise

    # create model using inferred node_num
    output_len = cfg.get('output_len', 1)
    layers = cfg.get('layers', 2)
    input_dims = cfg.get('input_dims', 8)
    node_dims = cfg.get('node_dims', 16)
    time_dims = cfg.get('time_dims', 4)

    model = PatchSTG(output_len=output_len, node_num=N, layers=layers, input_dims=input_dims, node_dims=node_dims, time_dims=time_dims)
    model.to(device)
    logging.info('Model initialized')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
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
            pred = model(x, te, mask=mask)  # [B, T, N, out_dim]
            # take last time step and first output dim
            pred_last = pred[:, -1, :, 0]  # [B, N]
            target = labels[..., 0]       # [B, N]
            loss = masked_mae(pred_last, target, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (i+1) % 10 == 0:
                logging.info(f'Epoch {epoch} batch {i+1} loss={loss.item():.6f}')

        dt = time.time() - t0
        avg_loss = epoch_loss / max(1, n_batches)
        logging.info(f'Epoch {epoch} finished: avg_loss={avg_loss:.6f} time={dt:.1f}s')

        # save checkpoint (ensure parent dir exists and log failures)
        ckpt = {'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        ckpt_path = cfg.get('ckpt_path', f'checkpoint_epoch{epoch}.pth')
        try:
            ckpt_dir = os.path.dirname(ckpt_path) or '.'
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(ckpt, ckpt_path)
            logging.info(f'Saved checkpoint to {ckpt_path}')
        except Exception:
            logging.exception(f'Failed to save checkpoint to {ckpt_path}; attempting fallback in cwd')
            fallback = f'checkpoint_epoch{epoch}_{int(time.time())}.pth'
            try:
                torch.save(ckpt, fallback)
                logging.info(f'Saved checkpoint to fallback {fallback}')
            except Exception:
                logging.exception(f'Failed to save fallback checkpoint {fallback}')

    logging.info('Training finished')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Ensure any uncaught exception is recorded to log file as well
        logging.exception('Uncaught exception in train.py')
        raise
