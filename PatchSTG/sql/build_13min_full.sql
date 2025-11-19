-- ********************************************************************--
-- 文件： sql/build_13min_full.sql
-- 作者： 撼风（为用户适配）
-- 目的： 从三张原始表（s_traffic_light_traj_flow_stat、traffic_light_node_basis、tb_inter_day_type）
--       生成按分钟的样本行，并为每个样本构造 13 分钟的输入序列（past12 + current）以及对应的时间特征。
-- 说明： 该脚本综合完成分钟聚合、24 段映射、上游映射、节点级 13min 拼接等步骤，输出最终样本表 `tb_samples_13min_final`。
-- 运行环境：MaxCompute (ODPS)。强烈建议先在小范围（单个 adcode、单日 ds）上测试每一步，
--       并确认目标环境对 WM_CONCAT、REGEXP_EXTRACT、POSEXPLODE 等函数的支持和行为。
-- 创建时间：2025-11-17
-- ********************************************************************--

USE autonavi_traffic_report;
SET odps.sql.joiner.instances=3000;

-- --------------------------------------------------------------------
-- Step 0: 聚合原始流量到每分钟粒度（agg_turns_minute）
-- --------------------------------------------------------------------
DROP TABLE IF EXISTS autonavi_traffic_report.agg_turns_minute PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.agg_turns_minute LIFECYCLE 31 AS
SELECT
  CAST(adcode AS BIGINT) AS adcode,
  CAST(nds_id AS BIGINT) AS nds_id,
  CAST(next_nds_id AS BIGINT) AS next_nds_id,
  ds,
  FROM_UNIXTIME(UNIX_TIMESTAMP(passts_time,'yyyy-MM-dd HH:mm:ss'),'yyyy-MM-dd HH:mm:00') AS minute_ts,
  SUM(cnt) AS cnt
FROM autonavi_traffic_report.s_traffic_light_traj_flow_stat
WHERE ds BETWEEN '20250701' AND '20251031'
GROUP BY CAST(adcode AS BIGINT), CAST(nds_id AS BIGINT), CAST(next_nds_id AS BIGINT), ds,
  FROM_UNIXTIME(UNIX_TIMESTAMP(passts_time,'yyyy-MM-dd HH:mm:ss'),'yyyy-MM-dd HH:mm:00')
;

-- --------------------------------------------------------------------
-- Step 1: 筛选交叉口（inter_filter）—— 保证样本只来自有信控路口定义的转向
-- --------------------------------------------------------------------
DROP TABLE IF EXISTS autonavi_traffic_report.inter_filter PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.inter_filter LIFECYCLE 31 AS
SELECT
  a.ds,
  CAST(a.nds_id AS BIGINT) AS nds_id,
  CAST(a.next_nds_id AS BIGINT) AS next_nds_id,
  CAST(a.adcode AS BIGINT) AS adcode,
  a.passts_time,
  a.cnt AS flow
FROM (
  SELECT ds, nds_id, next_nds_id, adcode, passts_time, cnt
  FROM autonavi_traffic_report.s_traffic_light_traj_flow_stat
  WHERE ds BETWEEN '20250701' AND '20251031'
) a
JOIN (
  SELECT DISTINCT nds_id, next_nds_id, adcode, ds
  FROM autonavi_traffic_report.traffic_light_node_basis
  WHERE ds BETWEEN '20250701' AND '20251031' AND next_nds_id > 0
) b
ON a.nds_id = b.nds_id AND a.next_nds_id = b.next_nds_id AND CAST(a.adcode AS BIGINT) = CAST(b.adcode AS BIGINT) AND a.ds = b.ds
;

-- --------------------------------------------------------------------
-- Step 2: 为每个样本生成 24 个 "映射时间"（maptime）对应前 1..24 分钟
--         生成表 spatial_map (nds_id,next_nds_id,adcode,ds,passts_time,flow_label,pos,maptime)
-- --------------------------------------------------------------------
DROP TABLE IF EXISTS autonavi_traffic_report.spatial_map PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.spatial_map LIFECYCLE 31 AS
SELECT
  nds_id,
  next_nds_id,
  adcode,
  ds,
  passts_time,
  flow AS flow_label,
  pos,
  DATEADD(passts_time, -1*(1+pos), 'mi') AS maptime
FROM (
  SELECT ds, nds_id, next_nds_id, adcode, passts_time, flow
  FROM autonavi_traffic_report.inter_filter
) a
LATERAL VIEW POSEXPLODE(SPLIT(RTRIM(REPEAT('0 ',24)), ' ')) b AS pos, val
;

-- --------------------------------------------------------------------
-- Step 3: 关联每个 maptime 的真实出流量（来自 agg_turns_minute），生成 spatial_detail
--         包含 pos (0->t-1 ... 23->t-24) 的 exit_cnt
--         同时补入 time 特征（week,hour,minute,day_type,day,month）来自 maptime 与 tb_inter_day_type
-- --------------------------------------------------------------------
DROP TABLE IF EXISTS autonavi_traffic_report.spatial_detail PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.spatial_detail LIFECYCLE 31 AS
SELECT
  a.nds_id,
  a.next_nds_id,
  a.adcode,
  a.ds,
  a.passts_time,
  a.flow_label,
  a.maptime,
  a.pos,
  CAST(COALESCE(b.cnt, 0) AS STRING) AS exit_cnt,
  CAST((DAYOFWEEK(a.maptime)-1) AS STRING) AS time_week,
  CAST(HOUR(a.maptime) AS STRING) AS time_hour,
  CAST(MINUTE(a.maptime) AS STRING) AS time_minute,
  CAST(COALESCE(t.day_type, '0') AS STRING) AS day_type,
  CAST((DAY(a.maptime)-1) AS STRING) AS time_day,
  CAST((MONTH(a.maptime)-1) AS STRING) AS time_month
FROM autonavi_traffic_report.spatial_map a
LEFT JOIN autonavi_traffic_report.agg_turns_minute b
  ON CAST(a.adcode AS BIGINT) = CAST(b.adcode AS BIGINT)
  AND a.maptime = b.minute_ts
  AND a.nds_id = b.nds_id
  AND a.next_nds_id = b.next_nds_id
LEFT JOIN autonavi_traffic_brain.tb_inter_day_type t
  ON a.ds = t.ds
WHERE a.ds BETWEEN '20250701' AND '20251031'
;

-- --------------------------------------------------------------------
-- Step 4: 将 spatial_detail 按 (nds_id,next_nds_id,adcode,ds,passts_time) 聚合成 pretrain 行
--         dym_feat_feat: 24 段 exit_cnt, ordered by pos ASC (pos=0 => t-1, pos=23 => t-24), 用 ';' 分隔
--         time_feat: 24 段 time token (week hour minute day_type day month) 每段用空格连接，段与段用 ';' 分隔
-- --------------------------------------------------------------------
DROP TABLE IF EXISTS autonavi_traffic_report.tb_inter_spatial_pretrain_data_temp PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.tb_inter_spatial_pretrain_data_temp LIFECYCLE 31 AS
SELECT
  nds_id,
  next_nds_id,
  adcode,
  ds,
  passts_time,
  flow_label,
  WM_CONCAT(exit_cnt) WITHIN GROUP (ORDER BY pos) AS dym_feat_feat,
  WM_CONCAT(CONCAT(time_week, ' ', time_hour, ' ', time_minute, ' ', day_type, ' ', time_day, ' ', time_month)) WITHIN GROUP (ORDER BY pos) AS time_feat
FROM autonavi_traffic_report.spatial_detail
GROUP BY nds_id, next_nds_id, adcode, ds, passts_time, flow_label
;

-- --------------------------------------------------------------------
-- Step 5: 构建上游映射表 tb_turn_upstream_map（当前转向的上游为所有 next_nds_id == 当前 nds_id）
-- --------------------------------------------------------------------
DROP TABLE IF EXISTS autonavi_traffic_report.tb_turn_upstream_map PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.tb_turn_upstream_map LIFECYCLE 31 AS
SELECT DISTINCT
  t_target.adcode AS adcode,
  t_target.ds AS ds,
  CAST(t_target.nds_id AS BIGINT) AS current_nds_id,
  CAST(t_target.next_nds_id AS BIGINT) AS current_next_nds_id,
  CAST(t_up.nds_id AS BIGINT) AS upstream_nds_id,
  CAST(t_up.next_nds_id AS BIGINT) AS upstream_next_nds_id
FROM autonavi_traffic_report.traffic_light_node_basis t_target
JOIN autonavi_traffic_report.traffic_light_node_basis t_up
  ON CAST(t_up.next_nds_id AS BIGINT) = CAST(t_target.nds_id AS BIGINT)
  AND t_up.adcode = t_target.adcode
  AND t_up.ds = t_target.ds
WHERE t_target.next_nds_id > 0
  AND t_up.next_nds_id > 0
  AND t_target.ds BETWEEN '20250701' AND '20251031'
  AND t_up.ds BETWEEN '20250701' AND '20251031'
;

-- --------------------------------------------------------------------
-- Step 6: 为每个 node (包括目标和其上游) 生成 13-min flows 字符串 (t-12,...,t)
--         先从 tb_inter_spatial_pretrain_data_temp 中，展开 dym_feat_feat pos<12（pos 0..11 => t-1..t-12），
--         再按 node/per passts_time 聚合并按 pos DESC 排序得到 t-12..t-1，然后在末尾拼接 flow_label
-- --------------------------------------------------------------------
DROP TABLE IF EXISTS autonavi_traffic_report.tb_node_13min_features PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.tb_node_13min_features LIFECYCLE 31 AS
SELECT
  p.nds_id,
  p.next_nds_id,
  p.adcode,
  p.ds,
  p.passts_time,
  p.flow_label,
  -- past12_csv: t-12,t-11,...,t-1 (comma separated)
  TRIM(BOTH ',' FROM WM_CONCAT(CONCAT(val, ',')) WITHIN GROUP (ORDER BY pos DESC)) AS past12_csv,
  -- flows13: t-12,...,t-1,current (comma separated)
  CONCAT(
    TRIM(BOTH ',' FROM WM_CONCAT(CONCAT(val, ',')) WITHIN GROUP (ORDER BY pos DESC)),
    ',',
    CAST(p.flow_label AS STRING)
  ) AS flows13,
  -- time13: t-12...t-1,current_time_token (segments separated by ';', per-segment token is the same space-joined string used in time_feat)
  CONCAT(
    TRIM(BOTH ';' FROM WM_CONCAT(CONCAT(val_time, ';')) WITHIN GROUP (ORDER BY pos DESC)),
    ';',
    -- current time token: take the first token of p.time_feat (pos 0) -> use REGEXP_EXTRACT
    REGEXP_EXTRACT(p.time_feat, '^([^;]*)')
  ) AS time13
FROM (
  -- explode dym_feat_feat into pos,val for pos<12
  SELECT
    nds_id, next_nds_id, adcode, ds, passts_time, flow_label,
    pos,
    val AS val,
    CONCAT(time_token) AS val_time
  FROM (
    SELECT nds_id, next_nds_id, adcode, ds, passts_time, flow_label, dym_feat_feat, time_feat
    FROM autonavi_traffic_report.tb_inter_spatial_pretrain_data_temp
  ) base
  LATERAL VIEW POSEXPLODE(SPLIT(dym_feat_feat, ';')) pe AS pos, val
  LATERAL VIEW POSEXPLODE(SPLIT(time_feat, ';')) te AS pos2, time_token
  WHERE pos = pos2 AND pos < 12
) exploded
GROUP BY nds_id, next_nds_id, adcode, ds, passts_time, flow_label
;

-- --------------------------------------------------------------------
-- Step 7: 最终样本表：对每个目标转向 (nds_id,next_nds_id,passts_time) 取其自身与上游节点，按节点顺序拼接 node_pairs, input_flows, time_features
--         node_pairs 格式： "nds_next;nds_next;..."
--         input_flows 格式： "nds_next:flow_t-12,flow_t-11,...,flow_t;nds_next:..."
-- --------------------------------------------------------------------
DROP TABLE IF EXISTS autonavi_traffic_report.tb_samples_13min_full PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.tb_samples_13min_full LIFECYCLE 100 AS
SELECT
  -- sample id: nds_next|passts_time
  CONCAT(CAST(t.nds_id AS STRING), '_', CAST(t.next_nds_id AS STRING), '|', t.passts_time) AS sample_id,
  CAST(t.adcode AS BIGINT) AS adcode,
  t.passts_time AS start_time,

  -- input_flows: per-node flows13 (comma separated minute values) with nodes separated by ';'
  COALESCE(WM_CONCAT(CONCAT(CONCAT(CAST(n.nds_id AS STRING), '_', CAST(n.next_nds_id AS STRING)), ':', n.flows13)) WITHIN GROUP (ORDER BY n.nds_id, n.next_nds_id), 'NONE') AS input_flows,

  -- time_features: per-node time13 (segments separated by ';')
  COALESCE(WM_CONCAT(CONCAT(CONCAT(CAST(n.nds_id AS STRING), '_', CAST(n.next_nds_id AS STRING)), ':', n.time13)) WITHIN GROUP (ORDER BY n.nds_id, n.next_nds_id), 'NONE') AS time_features,

  -- node_pairs
  COALESCE(WM_CONCAT(CONCAT(CAST(n.nds_id AS STRING), '_', CAST(n.next_nds_id AS STRING))) WITHIN GROUP (ORDER BY n.nds_id, n.next_nds_id), 'NONE') AS node_pairs,

  -- node_count
  COUNT(DISTINCT CONCAT(CAST(n.nds_id AS STRING), '_', CAST(n.next_nds_id AS STRING))) AS node_count,

  SUBSTR(t.passts_time, 1, 10) AS sample_date,
  SUBSTR(t.passts_time, 12, 5) AS sample_time_of_day

FROM (
  -- base target samples: use pretrain temp as the canonical list of target samples
  SELECT nds_id, next_nds_id, adcode, ds, passts_time
  FROM autonavi_traffic_report.tb_inter_spatial_pretrain_data_temp
) t
-- join target's own node features
LEFT JOIN autonavi_traffic_report.tb_node_13min_features n0
  ON n0.nds_id = t.nds_id AND n0.next_nds_id = t.next_nds_id AND n0.adcode = t.adcode AND n0.ds = t.ds AND n0.passts_time = t.passts_time
-- join upstream nodes
LEFT JOIN autonavi_traffic_report.tb_turn_upstream_map um
  ON um.current_nds_id = t.nds_id AND um.current_next_nds_id = t.next_nds_id AND um.adcode = t.adcode AND um.ds = t.ds
LEFT JOIN autonavi_traffic_report.tb_node_13min_features n
  ON n.nds_id = um.upstream_nds_id AND n.next_nds_id = um.upstream_next_nds_id AND n.adcode = um.adcode AND n.ds = um.ds AND n.passts_time = t.passts_time

-- To include the target node itself in the aggregation, UNION target node row with upstream rows via a derived table
-- For simplicity, we use a lateral approach: assemble rows from n0 (target) and n (upstreams) via UNION ALL in a subquery
;

-- Because MaxCompute does not directly support lateral union in the SELECT above, we create an intermediate table that
-- unifies target node + upstream nodes per target sample, then aggregate that into final samples table.

-- Step 7a: build per-sample node rows (target + upstream)
DROP TABLE IF EXISTS autonavi_traffic_report.tb_samples_node_rows PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.tb_samples_node_rows LIFECYCLE 31 AS
SELECT
  t.nds_id AS target_nds, t.next_nds_id AS target_next, t.adcode, t.ds, t.passts_time,
  n0.nds_id AS node_nds, n0.next_nds_id AS node_next, n0.flows13 AS flows13, n0.time13 AS time13
FROM autonavi_traffic_report.tb_inter_spatial_pretrain_data_temp t
LEFT JOIN autonavi_traffic_report.tb_node_13min_features n0
  ON n0.nds_id = t.nds_id AND n0.next_nds_id = t.next_nds_id AND n0.adcode = t.adcode AND n0.ds = t.ds AND n0.passts_time = t.passts_time

UNION ALL

SELECT
  t.nds_id AS target_nds, t.next_nds_id AS target_next, t.adcode, t.ds, t.passts_time,
  n.nds_id AS node_nds, n.next_nds_id AS node_next, n.flows13 AS flows13, n.time13 AS time13
FROM autonavi_traffic_report.tb_inter_spatial_pretrain_data_temp t
JOIN autonavi_traffic_report.tb_turn_upstream_map um
  ON um.current_nds_id = t.nds_id AND um.current_next_nds_id = t.next_nds_id AND um.adcode = t.adcode AND um.ds = t.ds
LEFT JOIN autonavi_traffic_report.tb_node_13min_features n
  ON n.nds_id = um.upstream_nds_id AND n.next_nds_id = um.upstream_next_nds_id AND n.adcode = um.adcode AND n.ds = um.ds AND n.passts_time = t.passts_time
;

-- Step 7b: aggregate node rows into final samples
DROP TABLE IF EXISTS autonavi_traffic_report.tb_samples_13min_final PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.tb_samples_13min_final LIFECYCLE 100 AS
SELECT
  CONCAT(CAST(target_nds AS STRING), '_', CAST(target_next AS STRING), '|', passts_time) AS sample_id,
  CAST(adcode AS BIGINT) AS adcode,
  passts_time AS start_time,
  COALESCE((WM_CONCAT(CONCAT(CONCAT(CAST(node_nds AS STRING), '_', CAST(node_next AS STRING)), ':', COALESCE(flows13, '0'))) WITHIN GROUP (ORDER BY node_nds, node_next)), 'NONE') AS input_flows,

  COALESCE((WM_CONCAT(CONCAT(CONCAT(CAST(node_nds AS STRING), '_', CAST(node_next AS STRING)), ':', COALESCE(time13, ''))) WITHIN GROUP (ORDER BY node_nds, node_next)), 'NONE') AS time_features,

  COALESCE((WM_CONCAT(CONCAT(CAST(node_nds AS STRING), '_', CAST(node_next AS STRING)) ) WITHIN GROUP (ORDER BY node_nds, node_next)), 'NONE') AS node_pairs,
  SUBSTR(passts_time,12,5) AS sample_time_of_day
FROM autonavi_traffic_report.tb_samples_node_rows
GROUP BY target_nds, target_next, adcode, ds, passts_time
;

-- Quick check: show some rows
SELECT * FROM autonavi_traffic_report.tb_samples_13min_final LIMIT 50;

-- ********************************************************************--
-- 注意与调试建议：
-- 1) 在你们的 ODPS 环境中，WM_CONCAT 和 REGEXP_EXTRACT 的行为可能略有差异，请先在小样本上跑通每一步（建议按单个 adcode, 单天 ds）
-- 2) 若 WM_CONCAT 不支持 CONCAT inside WITHIN GROUP，请改用 WM_CONCAT(CONCAT(...)) 或先用 GROUP_CONCAT（视环境可用函数）
-- 3) 如果你希望 flows13 内的分钟顺序严格为 t-12..t（远->近->current），本脚本已按 pos DESC 聚合来实现（pos 11..0），并在末尾拼接 current
-- 4) 若需把 input_flows 内节点顺序按特定策略（例如 target first, then upstream ordered by historical volume），我可以调整 ORDER BY 子句实现
-- ********************************************************************--
