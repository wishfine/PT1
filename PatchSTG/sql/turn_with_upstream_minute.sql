-- ********************************************************************--
-- 文件： sql/turn_with_upstream_minute.sql
-- 作者： 撼风（为用户适配）
-- 目的： 计算每个目标转向在分钟粒度的流量，并同时计算该目标的 1-hop 上游转向
--       在每分钟的合计流量与上游明细（用于后续构造上游特征）。
-- 说明： 该脚本包含：1) 每分钟聚合；2) 构建转向与上游映射；3) 计算上游合计与明细；4) 生成对齐后的表 `turn_with_upstream_minute`。
-- 运行环境：MaxCompute (ODPS)。请按需修改 ds 范围与生命周期参数，并在小范围验证函数支持情况。
-- 创建时间：2025-11-17
-- ********************************************************************--

-- 1) 每分钟聚合原始流量表（agg_turns_minute）
DROP TABLE IF EXISTS autonavi_traffic_report.agg_turns_minute PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.agg_turns_minute LIFECYCLE 31 AS
SELECT
  CAST(adcode AS BIGINT) AS adcode,
  CAST(nds_id AS BIGINT) AS nds_id,
  CAST(next_nds_id AS BIGINT) AS next_nds_id,
  ds,
  -- 将 passts_time 截断到分钟：'yyyy-MM-dd HH:mm:00'
  FROM_UNIXTIME(UNIX_TIMESTAMP(passts_time,'yyyy-MM-dd HH:mm:ss'),'yyyy-MM-dd HH:mm:00') AS minute_ts,
  SUM(cnt) AS cnt
FROM autonavi_traffic_report.s_traffic_light_traj_flow_stat
WHERE ds BETWEEN '20250701' AND '20251031'   -- <- 根据需要调整时间范围
GROUP BY CAST(adcode AS BIGINT), CAST(nds_id AS BIGINT), CAST(next_nds_id AS BIGINT), ds,
  FROM_UNIXTIME(UNIX_TIMESTAMP(passts_time,'yyyy-MM-dd HH:mm:ss'),'yyyy-MM-dd HH:mm:00')
;

-- 2) 构建“目标转向”与其 1-hop 上游的映射表（upstream_map）
-- 用 traffic_light_node_basis 自连接：上游行的 next_nds_id == 目标行的 nds_id
DROP TABLE IF EXISTS autonavi_traffic_report.upstream_map PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.upstream_map LIFECYCLE 31 AS
SELECT DISTINCT
  t_target.adcode AS adcode,
  t_target.ds AS ds,
  CAST(t_target.nds_id AS BIGINT) AS target_nds_id,
  CAST(t_target.next_nds_id AS BIGINT) AS target_next_nds_id,
  CAST(t_up.nds_id AS BIGINT) AS up_nds_id,
  CAST(t_up.next_nds_id AS BIGINT) AS up_next_nds_id
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

-- 3) 计算“上游到某节点 A 的每分钟合计流量” —— up_minute_sum
DROP TABLE IF EXISTS autonavi_traffic_report.up_minute_sum PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.up_minute_sum LIFECYCLE 31 AS
SELECT
  adcode,
  ds,
  next_nds_id AS target_nds_id,
  minute_ts,
  SUM(cnt) AS up_sum_cnt
FROM autonavi_traffic_report.agg_turns_minute
WHERE ds BETWEEN '20250701' AND '20251031'
GROUP BY adcode, ds, next_nds_id, minute_ts
;

-- 4) 生成“上游明细表”（每个上游转向在每分钟的流量）
DROP TABLE IF EXISTS autonavi_traffic_report.up_minute_detail PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.up_minute_detail LIFECYCLE 31 AS
SELECT
  a.adcode,
  a.ds,
  a.nds_id AS up_nds_id,
  a.next_nds_id AS up_next_nds_id,
  a.minute_ts,
  a.cnt AS up_cnt
FROM autonavi_traffic_report.agg_turns_minute a
WHERE a.ds BETWEEN '20250701' AND '20251031'
;

-- 5) 最终表：将目标转向与上游合计与明细对齐输出（turn_with_upstream_minute）
DROP TABLE IF EXISTS autonavi_traffic_report.turn_with_upstream_minute PURGE;
CREATE TABLE IF NOT EXISTS autonavi_traffic_report.turn_with_upstream_minute LIFECYCLE 31 AS
SELECT
  t.adcode,
  t.ds,
  t.target_nds_id AS nds_id,
  t.target_next_nds_id AS next_nds_id,
  COALESCE(a.minute_ts, up.minute_ts) AS minute_ts,
  COALESCE(a.cnt, 0) AS target_cnt,
  COALESCE(up.up_sum_cnt, 0) AS up_sum_cnt,
  -- upstream 明细：把所有上游转向 (up_nds_id->up_next_nds_id) 在该分钟的 cnt 用分号分隔拼接（可解析）
  WM_CONCAT(CONCAT(CONCAT(CAST(d.up_nds_id AS STRING), '->', CAST(d.up_next_nds_id AS STRING)), ' ', CAST(d.up_cnt AS STRING))) WITHIN GROUP(ORDER BY d.up_nds_id) AS upstream_detail
FROM (
  -- 目标转向集合（去重）
  SELECT DISTINCT CAST(adcode AS BIGINT) AS adcode, ds,
         CAST(nds_id AS BIGINT) AS target_nds_id, CAST(next_nds_id AS BIGINT) AS target_next_nds_id
  FROM autonavi_traffic_report.traffic_light_node_basis
  WHERE next_nds_id > 0
    AND ds BETWEEN '20250701' AND '20251031'
) t
LEFT JOIN autonavi_traffic_report.agg_turns_minute a
  ON a.adcode = t.adcode AND a.ds = t.ds
  AND a.nds_id = t.target_nds_id AND a.next_nds_id = t.target_next_nds_id
LEFT JOIN autonavi_traffic_report.up_minute_sum up
  ON up.adcode = t.adcode AND up.ds = t.ds AND up.target_nds_id = t.target_nds_id
  AND up.minute_ts = a.minute_ts
LEFT JOIN autonavi_traffic_report.up_minute_detail d
  ON d.adcode = t.adcode AND d.ds = t.ds AND d.up_next_nds_id = t.target_nds_id AND d.minute_ts = a.minute_ts
GROUP BY t.adcode, t.ds, t.target_nds_id, t.target_next_nds_id, COALESCE(a.minute_ts, up.minute_ts), COALESCE(a.cnt,0), COALESCE(up.up_sum_cnt,0)
ORDER BY t.adcode, t.target_nds_id, t.target_next_nds_id, minute_ts
;

-- 6) 快速检查样例
SELECT * FROM autonavi_traffic_report.turn_with_upstream_minute LIMIT 100;

-- 备注：
-- - 如果你希望保留 top-K 上游作为单独列 (up1_cnt, up2_cnt ...)，我可以在此基础上写一个额外查询来先按历史总流量对每个目标选 top-K，再在 minute 级别 join 出 up1/up2 列。
-- - 请在运行前根据实际 ds 范围和 MaxCompute 时间函数，做必要调整。
-- END
