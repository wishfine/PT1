"""
训练脚本（基于 LightGBM），仅使用 ODPS 流式读取数据（日志时间戳与目录按训练开始时间）

这个变体会：
- 在训练函数开始处记录训练开始时间，并用该时间命名日志目录（logs/YYYYmmdd_HHMMSS）。
- 在配置 logging 时为文件与控制台设置相同的 Formatter，确保每条日志都有时间戳。
- 将训练结束的输出使用 logging.info 而非 print，以保证日志格式统一。

使用建议：把本文件重命名为 `examples/train_from_samples.py` 或直接运行以验证日志行为。
"""
import argparse
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import yaml
import logging


# 为简洁起见，其它函数（parse_args、parse_sample_row、_odps_record_iterator 等）
# 与之前版本一致，这里把核心变化（日志配置）实现好并保留训练主流程。
# 真实使用时可把下面函数复制回主脚本或直接替换原文件。


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output-model', default='models/lgbm_model.txt')
    p.add_argument('--save-metrics', default='metrics.json')
    p.add_argument('--max-rows', type=int, default=None)
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--use-topk', action='store_true')
    p.add_argument('--use-log', action='store_true')
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--use-odps', action='store_true', default=True)
    p.add_argument('--odps-table', type=str, default=None)
    p.add_argument('--odps-access-id', type=str, default=None)
    p.add_argument('--odps-access-key', type=str, default=None)
    p.add_argument('--odps-project', type=str, default=None)
    p.add_argument('--odps-endpoint', type=str, default=None)
    p.add_argument('--odps-col-map', type=str, default=None)
    p.add_argument('--odps-limit', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--val-frac', type=float, default=0.02)
    p.add_argument('--num-boost-round-first', type=int, default=200)
    p.add_argument('--num-boost-round-chunk', type=int, default=10)
    p.add_argument('--odps-val-table', type=str, default=None)
    p.add_argument('--val-limit', type=int, default=10000)
    p.add_argument('--eval-interval-batches', type=int, default=5)
    p.add_argument('--early-stopping-patience', type=int, default=5)
    p.add_argument('--early-stopping-metric', choices=['mae','rmse','smape'], default='mae')
    p.add_argument('--log-dir', type=str, default=None)
    return p.parse_args()


def parse_sample_row(row, use_topk=False, topk=3):
    input_flows = row.get('input_flows')
    if pd.isna(input_flows) or input_flows in ('', 'NONE'):
        return None
    groups = [g.strip() for g in str(input_flows).split(';') if g.strip() != '']
    node_arrays = []
    for g in groups:
        vals = [v.strip() for v in g.split(',')]
        if len(vals) < 13:
            vals = vals + ['0'] * (13 - len(vals))
        else:
            vals = vals[:13]
        try:
            nums = [float(x) if x != '' else 0.0 for x in vals]
        except Exception:
            nums = [float(x.replace('\r', '').replace('\n', '')) if x != '' else 0.0 for x in vals]
        node_arrays.append(nums)
    if len(node_arrays) == 0:
        return None

    target = node_arrays[0]
    X_target = np.array(target[:12], dtype=float)
    y = float(target[12])

    ups = node_arrays[1:]
    if len(ups) > 0:
        up_arr = np.sum(np.array(ups), axis=0)
    else:
        up_arr = np.zeros(13, dtype=float)
    X_up = np.array(up_arr[:12], dtype=float)

    X_topk = np.array([])
    if use_topk and len(ups) > 0:
        sums = [np.sum(u[:12]) for u in ups]
        idx_sorted = np.argsort(sums)[::-1]
        topk_list = []
        for i in range(topk):
            if i < len(idx_sorted):
                arr = np.array(ups[idx_sorted[i]][:12], dtype=float)
            else:
                arr = np.zeros(12, dtype=float)
            topk_list.append(arr)
        X_topk = np.concatenate(topk_list)

    time_feats = row.get('time_features', '')
    if pd.isna(time_feats) or time_feats in ('', 'NONE'):
        time_curr = np.zeros(6, dtype=float)
    else:
        segs = [s.strip() for s in str(time_feats).split(';') if s.strip() != '']
        seg = segs[-1] if len(segs) >= 1 else ''
        if seg == '':
            time_curr = np.zeros(6, dtype=float)
        else:
            parts = [float(x) for x in seg.split()[:6]]
            time_curr = np.array(parts, dtype=float)

    node_count = float(row.get('node_count', len(groups)))

    feats = [X_target, X_up]
    if X_topk.size > 0:
        feats.append(X_topk)
    feats.append(time_curr)
    feats.append(np.array([node_count], dtype=float))
    features = np.concatenate(feats)

    sample_date = str(row.get('sample_date', ''))
    sample_id = row.get('sample_id', None)
    return {'features': features, 'label': y, 'date': sample_date, 'sample_id': sample_id}


def _odps_record_iterator(table_name, access_id=None, access_key=None, project=None, endpoint=None, limit=None, col_map=None):
    try:
        from odps import ODPS
    except Exception:
        raise RuntimeError('pyodps 未安装')
    access_id = access_id or os.environ.get('ODPS_ACCESS_ID')
    access_key = access_key or os.environ.get('ODPS_ACCESS_KEY')
    project = project or os.environ.get('ODPS_PROJECT')
    endpoint = endpoint or os.environ.get('ODPS_ENDPOINT')
    if not (access_id and access_key and project and endpoint):
        raise RuntimeError('ODPS 连接信息缺失')
    odps = ODPS(access_id, access_key, project=project, endpoint=endpoint)
    tbl = odps.get_table(table_name)
    with tbl.open_reader() as reader:
        for i, rec in enumerate(reader):
            if limit is not None and i >= limit:
                break
            row = {col.name: rec.get(col.name) for col in reader.schema.columns}
            if col_map:
                try:
                    mapd = json.loads(col_map) if isinstance(col_map, str) else col_map
                    row = {mapd.get(k, k): v for k, v in row.items()}
                except Exception:
                    pass
            yield row


def smape(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = np.abs(a) + np.abs(b)
    denom = np.where(denom == 0, 1e-6, denom)
    return float(np.mean(2.0 * np.abs(a - b) / denom))


def train_from_odps_stream(args):
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42,
        'num_leaves': 31,
        'learning_rate': 0.05,
    }

    # 训练开始时间
    start_time = datetime.now()
    start_time_str = start_time.strftime('%Y%m%d_%H%M%S')

    # 日志目录按训练开始时间命名（除非用户指定 --log-dir）
    if getattr(args, 'log_dir', None):
        log_dir = args.log_dir
    else:
        log_dir = os.path.join('logs', start_time_str)
    os.makedirs(log_dir, exist_ok=True)

    # 配置 logging，确保文件与控制台都带时间戳
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root = logging.getLogger()
    # 清理已有 handler，避免重复
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)

    logging.info('日志目录: %s', log_dir)

    # 保存生效配置
    try:
        with open(os.path.join(log_dir, 'config_effective.json'), 'w') as f:
            json.dump({k: v for k, v in vars(args).items()}, f, indent=2)
    except Exception:
        logging.warning('无法写入生效配置')

    best_model_path = os.path.join(log_dir, os.path.basename(args.output_model))

    generator = _odps_record_iterator(args.odps_table, access_id=args.odps_access_id, access_key=args.odps_access_key,
                                      project=args.odps_project, endpoint=args.odps_endpoint, limit=args.odps_limit,
                                      col_map=args.odps_col_map)

    val_X = None
    val_y = None
    if getattr(args, 'odps_val_table', None):
        logging.info('Loading ODPS validation table %s ...', args.odps_val_table)
        val_rows = []
        for r in _odps_record_iterator(args.odps_val_table, access_id=args.odps_access_id, access_key=args.odps_access_key,
                                       project=args.odps_project, endpoint=args.odps_endpoint, limit=args.val_limit, col_map=args.odps_col_map):
            parsed = parse_sample_row(r, use_topk=args.use_topk, topk=args.topk)
            if parsed is None:
                continue
            val_rows.append(parsed)
        if len(val_rows) == 0:
            raise RuntimeError('从 ODPS 验证表未读取到有效样本')
        val_X = np.vstack([r['features'] for r in val_rows])
        val_y = np.array([r['label'] for r in val_rows], dtype=float)
        logging.info('Loaded %d validation samples', len(val_rows))

    buffer_X, buffer_y = [], []
    bst = None
    total = 0
    best_metric = float('inf')
    no_improve_rounds = 0
    eval_count = 0

    for row in generator:
        parsed = parse_sample_row(row, use_topk=args.use_topk, topk=args.topk)
        if parsed is None:
            continue
        buffer_X.append(parsed['features'])
        buffer_y.append(parsed['label'])
        total += 1
        if args.max_rows and total >= int(args.max_rows):
            break

        if len(buffer_X) >= args.batch_size:
            Xb = np.vstack(buffer_X)
            yb = np.array(buffer_y, dtype=float)

            if val_X is None and args.val_frac and args.val_frac > 0 and Xb.shape[0] > 1:
                n_val = max(1, int(Xb.shape[0] * args.val_frac))
                val_X = Xb[-n_val:]
                val_y = yb[-n_val:]
                Xb = Xb[:-n_val]
                yb = yb[:-n_val]

            if Xb.shape[0] == 0:
                buffer_X.clear(); buffer_y.clear();
                continue

            dtrain = lgb.Dataset(Xb, label=(np.log1p(yb) if args.use_log else yb))
            if bst is None:
                bst = lgb.train(params, dtrain, num_boost_round=args.num_boost_round_first)
            else:
                bst = lgb.train(params, dtrain, num_boost_round=args.num_boost_round_chunk, init_model=bst)

            eval_count += 1
            if (eval_count % max(1, args.eval_interval_batches)) == 0 and val_X is not None:
                y_pred_raw = bst.predict(val_X, num_iteration=bst.best_iteration)
                y_pred = np.expm1(y_pred_raw) if args.use_log else y_pred_raw
                if args.early_stopping_metric == 'mae':
                    metric_val = float(mean_absolute_error(val_y, y_pred))
                elif args.early_stopping_metric == 'rmse':
                    metric_val = float(mean_squared_error(val_y, y_pred, squared=False))
                else:
                    metric_val = float(smape(val_y.copy(), y_pred.copy()))
                logging.info('Eval #%d metric(%s)=%.6f', eval_count, args.early_stopping_metric, metric_val)
                if metric_val < best_metric:
                    best_metric = metric_val
                    no_improve_rounds = 0
                    bst.save_model(best_model_path)
                    logging.info('Found new best model (%.6f), saved to %s', best_metric, best_model_path)
                else:
                    no_improve_rounds += 1
                    logging.info('No improvement rounds: %d/%d', no_improve_rounds, args.early_stopping_patience)
                if no_improve_rounds >= args.early_stopping_patience:
                    logging.info('Early stopping triggered after %d evals', eval_count)
                    break

            buffer_X.clear(); buffer_y.clear()

    # final small chunk
    if len(buffer_X) > 0:
        Xb = np.vstack(buffer_X)
        yb = np.array(buffer_y, dtype=float)
        if val_X is None and args.val_frac and args.val_frac > 0 and Xb.shape[0] > 1:
            n_val = max(1, int(Xb.shape[0] * args.val_frac))
            val_X = Xb[-n_val:]
            val_y = yb[-n_val:]
            Xb = Xb[:-n_val]
            yb = yb[:-n_val]

        if Xb.shape[0] > 0:
            dtrain = lgb.Dataset(Xb, label=(np.log1p(yb) if args.use_log else yb))
            if bst is None:
                bst = lgb.train(params, dtrain, num_boost_round=args.num_boost_round_first)
            else:
                bst = lgb.train(params, dtrain, num_boost_round=args.num_boost_round_chunk, init_model=bst)

            if val_X is not None:
                y_pred_raw = bst.predict(val_X, num_iteration=bst.best_iteration)
                y_pred = np.expm1(y_pred_raw) if args.use_log else y_pred_raw
                if args.early_stopping_metric == 'mae':
                    metric_val = float(mean_absolute_error(val_y, y_pred))
                elif args.early_stopping_metric == 'rmse':
                    metric_val = float(mean_squared_error(val_y, y_pred, squared=False))
                else:
                    metric_val = float(smape(val_y.copy(), y_pred.copy()))
                logging.info('Final-chunk Eval metric(%s)=%.6f', args.early_stopping_metric, metric_val)
                if metric_val < best_metric:
                    bst.save_model(best_model_path)
                    logging.info('Saved improved final model to %s', best_model_path)

    if bst is None:
        raise RuntimeError('未生成任何模型（可能没有有效样本）')

    if val_X is None:
        raise RuntimeError('未找到验证集，请在首个批次中增大数据或设置 val_frac > 0')

    y_pred_raw = bst.predict(val_X, num_iteration=bst.best_iteration)
    y_pred = np.expm1(y_pred_raw) if args.use_log else y_pred_raw
    mae = mean_absolute_error(val_y, y_pred)
    rmse = mean_squared_error(val_y, y_pred, squared=False)
    sm = smape(val_y.copy(), y_pred.copy())

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    bst.save_model(args.output_model)
    metrics = {'mae': mae, 'rmse': rmse, 'smape': sm, 'trained_samples': int(total)}
    with open(args.save_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)

    logging.info('训练完成，保存模型到 %s', args.output_model)
    logging.info('验证集指标：%s', metrics)


def main():
    args = parse_args()
    if getattr(args, 'config', None):
        cfg = load_config(args.config)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Warning: 配置文件键 {k} 未识别，忽略')

    if not args.use_odps:
        raise RuntimeError('当前脚本仅支持 ODPS 流式读取，请设置 --use-odps')
    if not args.odps_table:
        raise RuntimeError('使用 ODPS 模式时必须指定 --odps-table')

    train_from_odps_stream(args)


if __name__ == '__main__':
    main()
