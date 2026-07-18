# 可靠性数据集 → Nature Sensors 论文映射(2026-07-18)

数据源:`~/Desktop/磁毛传感器可靠性分析原始数据/`(含 2026-07-17/18 已处理的
`处理结果与图表_20260717/`,管线与图脚本齐全)。该数据集属于**另一篇可靠性
论文**(Belief Reliability),原始数据可复用,但:
- **图必须重新绘制**,不得与可靠性论文重复发表同一图;
- 两篇论文投稿时需相互引用并声明数据复用范围。

## 已填入论文的真实数字(来源可追溯)

| 数字 | 来源 | 填入位置 |
|---|---|---|
| 9 种几何 L∈{3,5,8}×d∈{0.5,0.8,1.0} | `PDMS_NdFeB/` 9 目录 | Results-2, Fig2 caption |
| 基体 = PDMS(非 Ecoflex) | 目录名/文档 | Results-2, Methods materials |
| M ≈ 82 kA/m(COMSOL 双案例标定,偏差<0.1%) | 处理说明 PaperFig2 节 | Results-2, Methods |
| 响应面 R²=0.92(切向)/0.68(法向),S≈3–50 mT/N,44 有效点 | `Fig2exp_*` + `Fig2exp_experimental_points.csv`(46 行) | Results-2, Fig2e caption |
| 重复性 CV=13.4%(20 试次,手动按压) | 处理说明 Fig4 | Results-2, Methods |
| 响应 25 ms / 恢复 26 ms(最快),中位响应 50 ms(81 事件) | `Fig8_all_events_stats.csv`(已抽验) | Results-2, Fig2f caption |
| 疲劳 150,000 次、n=3、灵敏度稳定 ±4%(50k→150k 可靠检查点) | `fatigue_sensitivity_vs_cycles.csv`(已抽验)+ Fig9 | Results-2, Fig2f caption |
| ATI F/T S/N FT35606,1000 Hz×16 平均=62.5 Hz 有效 | 力 CSV 文件头(已抽验) | Methods |
| 磁单传感器 ≈1100 Hz | 处理说明(≈1130 Hz) | Methods |
| 1920 对力-磁试次,1871 通过 QC(97.4%) | features_master.csv(1921 行含表头) | Methods |
| 疲劳驱动 0.52–0.62 N,检查点 5k–150k,每点~1000 循环 | fatigue csv + 处理说明 | Methods fatigue |

## 2026-07-18 第二轮:从 docx/xlsx 挖出的补充事实(已填入)

| 事实 | 来源 | 去向 |
|---|---|---|
| 阵列 pitch = 0.5 / 1.0 mm(2×2–5×5,L=3/5 mm),传感器距阵列中心下方 ≈1 mm | 0511多根磁毛实验验证.docx | Methods(数据集拆分为几何矩阵 + 阵列两类) |
| 理论模型 = 偶极子叠加 + 大变形梁;单毛理论-实验相对误差 0.265–0.409% | 0413法向力推导.docx + Belief docx | Methods |
| 疲劳实验 2026-06-23~26 执行,检查点排期与数据吻合,责任人李汶析 | 工作记录 xlsx | (元数据,备查) |
| NdFeB 退磁阈值 ~220°C(文献结论,docx 引用) | 磁毛退化实验方案.docx | 未填(非本文测量) |

**修正**:此前 Methods 把 1920 试次误归入 9 几何数据集,已拆分:
(i) 几何矩阵 = PDMS_NdFeB 9 组合;(ii) 阵列数据集 = 2×2–5×5(1920 试次)。

## 待向实验执行人确认(李汶析/李丽薇)

1. NdFeB 牌号、粒径/目数、与 PDMS 的质量配比(wt%)
2. PDMS 牌号与 base:curing 配比、固化温度与时间、真空脱泡参数
3. 预取向磁场与最终磁化场强度/方向
4. PDMS_NdFeB 9 组合样品的纤毛间距(阵列数据的 0.5/1.0 mm 不一定适用)
5. 形貌照片对应的显微镜型号与标定

## 仍未解决(占位符保留)

- **flat film / short pillar / non-magnetic / bare sensor 对照** → cilia gain、
  LOD 无法计算,`cilia-enhanced` 门控仍关闭(SOP-2)。
- 手动加载深度不可控(CV 13.4%)→ 正式 Fig. 2e/3c 的力标定建议按 SOP-1 用
  xArm 重做;Methods 已声明 robot 协议将取代该数据集的力标定口径。
- pitch 实测值、NdFeB 牌号/wt%、PDMS 配比、固化参数 → 查
  `磁毛传感器可靠性实验工作记录.xlsx`(本机无 openpyxl,未读)。
- 30 min 漂移、频率扫描仍缺。
- 2×2 传感器阵列反演(Fig. 3d/e)与 44 节点(Fig. 4)仍无数据——该数据集
  全部为单传感器 + 多纤毛 patch。

## 候选图(需要重绘后方可用于本论文)

- `figures_paper/Fig2exp_response_surface_PDMSNdFeB.pdf` → 本论文 Fig. 2e 原型
- `figures_paper/Fig9_fatigue_sensitivity_degradation.pdf` → 本论文 Fig. 2f 耐久 panel 原型
- `figures/Fig8_response_recovery.png` → Fig. 2f 动态 panel 原型
- `figures/Fig2/3/4/5/7` → 力-磁对齐、重复性、ΔB–F、切向对应 → Fig. 2/3 素材
- `形貌照片/P-S01..03` → Fig. 2c 实物照片来源
