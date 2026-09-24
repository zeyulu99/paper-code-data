# 图 2b 单列素材（2026-09-23）

脚本：`code/python/rebuild_fig2b_contrast.py`（单列）、`build_fig2b_panel_variants.py`（整块预览）
曲线读自已发布的面板数据 `outputs/figures_final/figure2_placement_ready/data/fig2b_*_T10.csv`，未重新计算。

## 两版的区别

| | v2_reference | v3_annotated |
|---|---|---|
| 每级输入的灰色虚线（C_f 后为 C_in，S 后为 C_f） | 有 | 有 |
| 输入峰值竖线、各曲线峰值圆点 | 有 | 有 |
| 幅值保持 A 与峰值延迟 Δt | 无 | 有（相对 C_in 的累计值） |

## 单列文件（三列各一张，可直接替换）

```
png/ pdf/ svg/
  fig2b_plug_flow_T10_v2_reference        fig2b_plug_flow_T10_v3_annotated
  fig2b_distributed_RTD_T10_v2_reference  fig2b_distributed_RTD_T10_v3_annotated
  fig2b_well_mixed_T10_v2_reference       fig2b_well_mixed_T10_v3_annotated
```

- 画布 2.55 × 2.75 in，坐标范围 (-0.08, 3.5)，与 figs.pptx 第 9 页现有的三张图完全一致，
  在 PPT 中用"更改图片 → 来自文件"替换即可，位置和大小不变。
- PNG 为 600 dpi、背景透明；PDF 与 SVG 为矢量，文字可编辑。
- 图中不含列标题、行标签（C_in/C_f/S）和时间轴箭头，这些保留在 PPT 中。
- 替换时注意列的对应：plug_flow、distributed_RTD、well_mixed 要与列标题一致。

## 整块预览（仅供比较，不用于排版）

`png/fig2b_panel_v2_reference.png`、`png/fig2b_panel_v3_annotated.png`

## 数值（相对 C_in）

| 拓扑 | C_f: A, Δt | S: A, Δt |
|---|---|---|
| Plug flow | 1.00, +5.0 min | 0.98, +6.0 min |
| Distributed RTD | 0.84, +4.6 min | 0.83, +5.6 min |
| Well mixed | 0.74, +3.3 min | 0.73, +4.3 min |

参数：T = 10 min 脉冲，τ_f = 5 min，τ_s = 1 min，gamma k = 3。

## 选定后需要同步的图注

在图 2 图注 (b) 中补两句：虚线为该级的输入；采用 v3 时另注明 A 与 Δt 是相对 C_in 的累计值。
