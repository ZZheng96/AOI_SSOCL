# stats null / 即时学习 consumption / UI 标签 进度

- [start] 建立进度文件
- [A done] 缺失值 None：routes_wo_stats(_p95 空→None、report summary defect_rate/avg/p95)、_wo_shared(bad_rate、stats.rate)、routes_detect(mean_latency_ms)、dashboard_page kpi_rate 显示“—”；routes_stats None/e2e 无 coalesce 已在工作区；旧测试无断言 0.0 需改；遗留：工单 report latencies 仍 e2e→latency_ms 回退（C 后补修：report 仅取 e2e 真实值、新增 n_e2e_missing，CSV 标注 e2e）
- [B done] service.submit_feedback_record 即时学习路径（legacy online，调用 engine.submit_feedback）：提交时取 consumable_regions 快照，persist_result 终态成功时调 _record_region_consumption(candidate_id=instant:feedback:{id}, kind=instant) 回写 n_tiles.region_review.consumption，trace.region_consumption 记状态；_record_region_consumption 增加 candidate_id/kind 可选参数（默认不变）
- [C done] region_review_dialog：新增 consumption_text（stale_after_use→“已用后变更” 优先于 used→“已用于学习”，否则“未用于学习”），表格末尾新增第 9 列“学习消费”（index 8，旧列索引不变）
