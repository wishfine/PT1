"""
训练脚本（基于 LightGBM），仅使用 ODPS 流式读取数据

说明：
- 本脚本通过 pyodps 从 ODPS 表逐行流式读取样本，逐条解析为特征向量与标签，并按 batch_size 聚合为训练块进行增量训练。
- 样本期望包含字段（示例）：sample_id, input_flows, time_features, node_count, sample_date 等。
- input_flows 格式：每个节点的 13 分钟序列（t-12,...,t），节点间用 ';' 分隔；第一个节点为目标转向，其余为上游转向。

设计要点：
- 强制使用 ODPS 流式读取（--use-odps），不再依赖 CSV 文件读取。
- 首个批次用较多的 boosting 轮数初始化 LightGBM 模型，后续批次在已有模型上使用较少轮数继续训练（init_model 增量训练）。
- 在首个批次中抽取少量作为验证集（val_frac），用于监控指标与基本评估。

使用示例（测试时先限制 odps_limit / max_rows）：
    export ODPS_ACCESS_ID=... ODPS_ACCESS_KEY=... ODPS_PROJECT=... ODPS_ENDPOINT=...
    python examples/train_from_samples.py --use-odps --odps-table your_project.your_table --odps-limit 5000 --batch-size 2048

注意：建议先在单个 adcode / 单日 ds 的小数据上逐步跑通每一步，并检查 ODPS 环境中 WM_CONCAT、POSEXPLODE、REGEXP_EXTRACT 等函数的行为兼容性。
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


def parse_args():
    p = argparse.ArgumentParser()
    # 输出与保存配置
    p.add_argument('--output-model', default='models/lgbm_model.txt', help='保存训练好模型的路径')
    p.add_argument('--save-metrics', default='metrics.json', help='保存评估指标的 JSON 文件路径')
    p.add_argument('--max-rows', type=int, default=None, help='调试时限制最多读取的样本数（仅作快速测试用途）')
    # 特征/模型选项
    p.add_argument('--topk', type=int, default=3, help='Top-K 的 K 值（当 --use-topk 启用时有效）')
    p.add_argument('--use-topk', action='store_true', help='是否把每个样本的 top-K 上游单独展开为特征')
    p.add_argument('--use-log', action='store_true', help='是否对标签使用 log1p 变换进行训练（预测时反变换）')
    p.add_argument('--split-method', choices=['date','stratified_time'], default='stratified_time', help='可选的时间切分方法（部分评估场景可用）')
    p.add_argument('--config', type=str, default=None, help='YAML/JSON 配置文件路径（可覆盖默认参数）')

    # ODPS 流式读取（本脚本核心）
    p.add_argument('--use-odps', action='store_true', default=True, help='是否使用 ODPS 流式读取（默认 True）')
    p.add_argument('--odps-table', type=str, default=None, help='要读取的 ODPS 表名，例如 project.table')
    p.add_argument('--odps-access-id', type=str, default=None, help='ODPS Access ID，可通过环境变量 ODPS_ACCESS_ID 提供')
    p.add_argument('--odps-access-key', type=str, default=None, help='ODPS Access Key，可通过环境变量 ODPS_ACCESS_KEY 提供')
    p.add_argument('--odps-project', type=str, default=None, help='ODPS project，可通过环境变量 ODPS_PROJECT 提供')
    p.add_argument('--odps-endpoint', type=str, default=None, help='ODPS endpoint，可通过环境变量 ODPS_ENDPOINT 提供')
    p.add_argument('--odps-col-map', type=str, default=None, help='可选的列名映射 JSON 字符串，用于把 ODPS 列名映射为脚本期望的列名')
    p.add_argument('--odps-limit', type=int, default=None, help='用于测试时限制从 ODPS 读取的总行数（None 表示不限）')

    # 流式训练控制参数
    p.add_argument('--batch-size', type=int, default=2048, help='流式读取并训练时的批次大小（每次构建一个训练块）')
    p.add_argument('--val-frac', type=float, default=0.02, help='从首个批次中抽取作为验证集的比例（用于 early stopping / 指标监控）')
    p.add_argument('--num-boost-round-first', type=int, default=200, help='用于首个批次的 boosting 轮数（初始化模型）')
    p.add_argument('--num-boost-round-chunk', type=int, default=10, help='后续每个批次用于继续训练的 boosting 轮数')
    # 验证/测试表与早停参数
    p.add_argument('--odps-val-table', type=str, default=None, help='可选：ODPS 验证表名（project.table），若提供将用于周期性评估）')
    p.add_argument('--odps-test-table', type=str, default=None, help='可选：ODPS 测试表名（project.table），训练结束后用于最终评估）')
    p.add_argument('--val-limit', type=int, default=10000, help='当使用 odps-val-table 时，最多读取的验证行数')
    p.add_argument('--eval-interval-batches', type=int, default=5, help='每多少个训练批次进行一次验证评估（默认 5）')
    p.add_argument('--early-stopping-patience', type=int, default=5, help='验证指标在多少次评估内无提升则提前停止（默认 5）')
    p.add_argument('--early-stopping-metric', choices=['mae','rmse','smape'], default='mae', help='用于早停的指标')
    p.add_argument('--log-dir', type=str, default=None, help='训练日志目录；如果未指定，脚本会在当前目录创建 logs/YYYYmmdd_HHMMSS')

    return p.parse_args()


def parse_sample_row(row, use_topk=False, topk=3):
    """解析一条样本行为模型输入特征与标签。

    输入：row（dict 或 类似 mapping 对象），字段例如 'input_flows','time_features','node_count','sample_date'。
    返回：字典，包含：
        - 'features': np.array 数值型特征向量（包含 target past12、upstream sum past12、可选 top-K、time token、node_count）
        - 'label': 浮点型标签（当前分钟流量 t）
        - 'date': 样本日期字符串（用于按日切分）
        - 'sample_id': 如存在则返回样本 id
    若样本无法解析或缺失 input_flows，则返回 None 表示跳过该条记录。
    """
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

    # target is first group
    target = node_arrays[0]
    X_target = np.array(target[:12], dtype=float)
    y = float(target[12])

    # upstream sum
    ups = node_arrays[1:]
    if len(ups) > 0:
        up_arr = np.sum(np.array(ups), axis=0)
    else:
        up_arr = np.zeros(13, dtype=float)
    X_up = np.array(up_arr[:12], dtype=float)

    # top-K upstream individual past12 sequences
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

    # parse time features: take last segment
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

    # node count
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


def stratified_time_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    分层时间划分：保证训练/验证/测试集都包含不同的星期和时段
    策略（简化实现）：针对每个 ISO 周分配日期到 train/val/test，优先把周末放到测试集
    返回：train_idx_mask, val_idx_mask, test_idx_mask (boolean masks for df.index)
    """
    df2 = df.copy()
    df2['date_dt'] = pd.to_datetime(df2['sample_date'], format='%Y%m%d')
    df2['dayofweek'] = df2['date_dt'].dt.dayofweek
    df2['week'] = df2['date_dt'].dt.isocalendar().week

    train_dates = []
    val_dates = []
    test_dates = []
    for week, week_df in df2.groupby('week'):
        dates = list(week_df['sample_date'].unique())
        # compute dayofweek per date
        dow_map = {d: int(pd.to_datetime(d, format='%Y%m%d').dayofweek) for d in dates}
        weekend = [d for d in dates if dow_map[d] >= 5]
        weekday = [d for d in dates if dow_map[d] < 5]
        # test: 2 days (prefer weekend)
        test = weekend[:2] if len(weekend) >= 2 else (weekend + weekday[:2-len(weekend)])
        remaining = [d for d in dates if d not in test]
        val = [remaining[len(remaining)//2]] if len(remaining) > 0 else []
        train = [d for d in remaining if d not in val]
        train_dates.extend(train)
        val_dates.extend(val)
        test_dates.extend(test)

    train_mask = df2['sample_date'].isin(train_dates).values
    val_mask = df2['sample_date'].isin(val_dates).values
    test_mask = df2['sample_date'].isin(test_dates).values
    return train_mask, val_mask, test_mask


def smape(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = np.abs(a) + np.abs(b)
    # avoid division by zero
    denom = np.where(denom == 0, 1e-6, denom)
    return float(np.mean(2.0 * np.abs(a - b) / denom))


def load_from_odps(table_name, access_id=None, access_key=None, project=None, endpoint=None, limit=10000, col_map=None):
    """Attempt to load rows from an ODPS table into a pandas.DataFrame.

    This function tries to import the pyodps SDK and will raise an informative
    error if it's not available. The returned DataFrame should match the CSV
    column expectations (sample_id, input_flows, time_features, node_count, sample_date, ...).

    col_map (dict) can be used to map ODPS column names to expected CSV column names.
    """
    try:
        from odps import ODPS
    except Exception as e:
        raise RuntimeError(
            'pyodps is required for --use-odps but not installed. Install with `pip install -U pyodps` or run with CSV input.'
        )

    # read credentials from args or environment
    access_id = access_id or os.environ.get('ODPS_ACCESS_ID')
    access_key = access_key or os.environ.get('ODPS_ACCESS_KEY')
    project = project or os.environ.get('ODPS_PROJECT')
    endpoint = endpoint or os.environ.get('ODPS_ENDPOINT')

    if not (access_id and access_key and project and endpoint):
        raise RuntimeError('ODPS credentials/connection info missing. Provide via args or environment variables ODPS_ACCESS_ID, ODPS_ACCESS_KEY, ODPS_PROJECT, ODPS_ENDPOINT')

    odps = ODPS(access_id, access_key, project=project, endpoint=endpoint)

    # Accept several table-name formats; try to extract table name
    tbl_name = table_name
    try:
        tbl = odps.get_table(tbl_name)
    except Exception:
        # try stripping common prefixes
        if tbl_name.startswith('odps://'):
            # user provided full URI, attempt naive split
            tbl_name = tbl_name.split('/')[-1]
        try:
            tbl = odps.get_table(tbl_name)
        except Exception as e:
            raise RuntimeError(f'Failed to get ODPS table {table_name}: {e}')

    rows = []
    # open_reader yields records; convert to dict and collect
    try:
        with tbl.open_reader() as reader:
            for i, rec in enumerate(reader):
                if i >= limit:
                    break
                # rec is a Record object; convert to dict by column names
                row = {col.name: rec.get(col.name) for col in reader.schema.columns}
                rows.append(row)
    except Exception as e:
        raise RuntimeError(f'Error reading rows from ODPS table {tbl_name}: {e}')

    if len(rows) == 0:
        raise RuntimeError('No rows read from ODPS table (empty or permission issue)')

    df = pd.DataFrame(rows)

    # optional column mapping
    if col_map:
        import json
        if isinstance(col_map, str):
            try:
                col_map = json.loads(col_map)
            except Exception:
                raise RuntimeError('Invalid JSON for --odps-col-map')
        df = df.rename(columns=col_map)

    return df


def load_config(path):
    """Load YAML or JSON config file and return a dict."""
    if path is None:
        return {}
    if path.lower().endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    # try yaml then json
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                return cfg
    except Exception:
        pass
    # fallback to json
    with open(path, 'r') as f:
        return json.load(f)


def _odps_record_iterator(table_name, access_id=None, access_key=None, project=None, endpoint=None, limit=None, col_map=None):
    """从 ODPS 表中逐行读取记录的生成器（streaming）。

    返回每一行的 dict（列名->值）。支持可选的 col_map 来重命名列（ODPS 列名 -> 脚本所需列名）。
    limit 为读取总行数上限（用于测试），None 表示不限制。
    """
    try:
        from odps import ODPS
    except Exception:
        raise RuntimeError('pyodps 未安装，请使用 `pip install pyodps` 安装后重试')

    access_id = access_id or os.environ.get('ODPS_ACCESS_ID')
    access_key = access_key or os.environ.get('ODPS_ACCESS_KEY')
    project = project or os.environ.get('ODPS_PROJECT')
    endpoint = endpoint or os.environ.get('ODPS_ENDPOINT')

    if not (access_id and access_key and project and endpoint):
        raise RuntimeError('ODPS 连接信息缺失，请通过参数或环境变量提供 ODPS_ACCESS_ID/ODPS_ACCESS_KEY/ODPS_PROJECT/ODPS_ENDPOINT')

    odps = ODPS(access_id, access_key, project=project, endpoint=endpoint)

    tbl_name = table_name
    try:
        tbl = odps.get_table(tbl_name)
    except Exception:
        if tbl_name.startswith('odps://'):
            tbl_name = tbl_name.split('/')[-1]
        tbl = odps.get_table(tbl_name)

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
                    # ignore mapping errors and yield original
                    pass
            yield row


def train_from_odps_stream(args):
    """主训练函数：从 ODPS 流式读取样本并增量训练 LightGBM。

    策略：按批次读取并解析为特征/标签；首个批次用于初始化模型（较多轮），后续批次在该模型上继续训练（较少轮）。
    """
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }

    # 记录训练开始时间，用于日志目录命名
    start_time = datetime.now()
    start_time_str = start_time.strftime('%Y%m%d_%H%M%S')

    # prepare log dir and save effective config if provided (do this early so logging is configured before any logging calls)
    if getattr(args, 'log_dir', None):
        log_dir = args.log_dir
    else:
        log_dir = os.path.join('logs', start_time_str)
    os.makedirs(log_dir, exist_ok=True)

    # configure logging with timestamped formatter for both file and console
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    # remove existing handlers to avoid duplicate logs in some environments
    if root_logger.handlers:
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info('日志目录: %s', log_dir)

    # 保存最终生效配置，便于复现
    try:
        with open(os.path.join(log_dir, 'config_effective.json'), 'w') as f:
            json.dump({k: v for k, v in vars(args).items()}, f, indent=2)
    except Exception:
        logging.warning('无法写入生效配置到日志目录')

    generator = _odps_record_iterator(
        args.odps_table,
        access_id=args.odps_access_id,
        access_key=args.odps_access_key,
        project=args.odps_project,
        endpoint=args.odps_endpoint,
        limit=args.odps_limit,
        col_map=args.odps_col_map,
    )

    # 如果提供了 ODPS 验证表，提前载入验证集（可按需限制行数）
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

    buffer_X = []
    buffer_y = []
    bst = None
    total = 0
    best_metric = float('inf')
    no_improve_rounds = 0
    eval_count = 0

    # NOTE: logging already configured earlier; best_model_path set above

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

            # 在首个批次中抽取部分作为验证集
            if val_X is None and args.val_frac and args.val_frac > 0 and Xb.shape[0] > 1:
                n_val = max(1, int(Xb.shape[0] * args.val_frac))
                val_X = Xb[-n_val:]
                val_y = yb[-n_val:]
                Xb = Xb[:-n_val]
                yb = yb[:-n_val]

            if Xb.shape[0] == 0:
                # 如果全部被划为验证集，清空 buffer 并继续
                buffer_X.clear(); buffer_y.clear();
                continue

            dtrain = lgb.Dataset(Xb, label=(np.log1p(yb) if args.use_log else yb))

            if bst is None:
                bst = lgb.train(params, dtrain, num_boost_round=args.num_boost_round_first, verbose_eval=False)
            else:
                bst = lgb.train(params, dtrain, num_boost_round=args.num_boost_round_chunk, init_model=bst, verbose_eval=False)

            eval_count += 1
            # 周期性评估
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
                # 判断是否提升（越小越好）
                if metric_val < best_metric:
                    best_metric = metric_val
                    no_improve_rounds = 0
                    # 保存最优模型
                    bst.save_model(best_model_path)
                    logging.info('Found new best model (%.6f), saved to %s', best_metric, best_model_path)
                else:
                    no_improve_rounds += 1
                    logging.info('No improvement rounds: %d/%d', no_improve_rounds, args.early_stopping_patience)
                # 触发早停
                if no_improve_rounds >= args.early_stopping_patience:
                    logging.info('Early stopping triggered after %d evals', eval_count)
                    break

            # 清空 buffer
            buffer_X.clear(); buffer_y.clear()

    # 处理残余小批次
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
                bst = lgb.train(params, dtrain, num_boost_round=args.num_boost_round_first, verbose_eval=False)
            else:
                bst = lgb.train(params, dtrain, num_boost_round=args.num_boost_round_chunk, init_model=bst, verbose_eval=False)

            # 对残余小批次也可以触发一次评估
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

    # 评估：使用 val_X/val_y（若存在），否则使用最后一个批次作为近似测试集
    if val_X is None:
        raise RuntimeError('未找到验证集，请在首个批次中增大数据或设置 val_frac > 0')

    y_pred_raw = bst.predict(val_X, num_iteration=bst.best_iteration)
    y_pred = np.expm1(y_pred_raw) if args.use_log else y_pred_raw
    mae = mean_absolute_error(val_y, y_pred)
    rmse = mean_squared_error(val_y, y_pred, squared=False)
    sm = smape(val_y.copy(), y_pred.copy())

    # 保存模型与指标
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    bst.save_model(args.output_model)
    metrics = {'mae': mae, 'rmse': rmse, 'smape': sm, 'trained_samples': int(total)}
    with open(args.save_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)

    print('训练完成，保存模型到', args.output_model)
    print('验证集指标：', metrics)


def main():
    args = parse_args()
    if getattr(args, 'config', None):
        cfg = load_config(args.config)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                print(f'Warning: 配置文件键 {k} 未识别，忽略')

    if not args.use_odps:
        raise RuntimeError('当前脚本仅支持 ODPS 流式读取，请设置 --use-odps')
    if not args.odps_table:
        raise RuntimeError('使用 ODPS 模式时必须指定 --odps-table')

    train_from_odps_stream(args)


if __name__ == '__main__':
    main()
