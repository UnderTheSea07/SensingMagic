# 图片资产清点与放置方案(2026-07-18,基于用户提供的 19 张图)

## ⚠️ 先说结论:Nature 政策红线

**Springer Nature / Nature Portfolio 不允许 AI 生成的图片用于发表**
(版权与研究诚信;仅"文章本身研究 AI"个案豁免)。
来源:nature.com/nature-portfolio/editorial-policies/ai。
→ 6 张 AI 概念图**只能作为排版蓝图**,终稿必须:
- 示意类 panel 用 Illustrator/Affinity 人工矢量重绘,或用真实 CAD 渲染;
- 数据类 panel 全部用真实数据重新生成(本仓库 analysis 脚本体系)。

## 一、AI 概念图 6 张(排版蓝图,不可直接投稿)

排版质量很高,叙事结构与我们的 Fig 1–5 规划高度一致,作为蓝图非常有价值。
但**其中所有"数据"均为虚构**,以下数字绝不能进入论文:

| AI 图 | 虚构内容(必须剔除/以真实数据替换) |
|---|---|
| Fig 1 mock | 概念性内容为主,风险最低;44 传感器数为设计意图(可保留为规划) |
| Fig 2 mock | ①3/5/7 wt% 应力-应变曲线及 E=1.02–1.72 MPa、UTS 表(真实数据是 0.68–0.85 MPa,见下);②"cilia-amplified ~5.1×"(门控未解锁);③F–\|Bz\| 扫场曲线;④"10⁴ cycles"(真实已有 1.5×10⁵,反而更强);⑤"RMS noise 0.32 mN" |
| Fig 3 mock | 定位误差曲线、力误差 30%→7%、SSIM 0.5–0.9、"bubble 0.6 mg / 0.05–0.2 mN"、touch/release ΔBz 曲线——**2×2 阵列尚无任何数据** |
| Fig 4 mock | 44 点布局(fingertip16/pad16/palm8/thumb4)、实时热图、known-position 验证——无数据;布局数字需以真实 CAD 为准 |
| Fig 5 mock | 风速标定 ΔB=0.089v²+0.528v+0.602 / R²=0.984 / RMSE 0.16 µT、动态阶梯、脉搏 CC=0.92 / RMSE 0.11、腕部空间图、横幅"≤5 m/s, t90<1s"——全部虚构,且与我们"风数据不足"的判定直接矛盾 |
| 五图总览 poster | "rise 12 ms / recovery 28 ms"、"σ=0.32 µT"、"10,000 cycles"等混合虚构数字 |

另有 1 张卡通机器人概念画:仅适合 TOC graphic / 学术海报,同样受 AI 政策约束。

## 二、真实图片(可用,按归属放置)

| 真实资产 | 放置 | 备注 |
|---|---|---|
| 制备流程示意(Ecoflex, 真空15min→热5min→浇注→磁预取向→热60min→脱模→强磁场) | **Fig. 2b** | 团队自制示意;需重绘成英文矢量统一风格。**推翻了"单步充磁"的理解——预取向确实存在**,Methods 已改回 |
| 纤毛 patch 实拍 ×3(弯曲阵列、透明基底) | **Fig. 2c** | 直接可用,需加 scale bar |
| 应力-应变(Non/x/y/z-mag,E_t=0.722/0.845/0.727/0.684±0.1 MPa) | **Fig. 2d** | 真实数据!数值已填入正文;建议拿原始数据按统一风格重绘,n 待确认 |
| Multi-Signal Alignment(力-磁对齐单事件) | Fig. 3 素材 | 已在重绘管线中用 aligned CSV 重画 |
| 磁体距离响应(20/30/40/50 mm,±40 mT) | Supplementary | 外加磁场对比实验,可作干扰表征 |
| 方波 XYZ 响应 + 0.6 s 上升/下降标注图 | Fig. 2f 素材 | ⚠️ 0.6 s 是**手动加载沿时长**,不是传感器响应时间(高速率敲击数据为 25–50 ms)——论文中不得混用 |
| 整手爆炸 CAD 渲染(Magnetic Array/Soft Elastomer/Magnetometer Array/FPCB) | **Fig. 1b / 4a** | 真实设计渲染,可用;与 AI 版风格接近但此为团队资产 |
| 风洞示意(modular low-speed wind tunnel + anemometer) | **Fig. 5a** | 团队自制,诚实(含 reference anemometer),可用 |
| 腕部佩戴示意 | **Fig. 5d** | 可用,需统一风格 |
| 脉搏波形(64.2 bpm, Bz) | Fig. 5d 素材 | 与我们 pulse pipeline 结果一致;建议用管线重绘统一风格 |

## 三、正在重绘的数据图(工作流 wf_0e2920fa)

全部由真实 CSV 生成、经对抗审查:
1. `fig3c_event_alignment` — 单事件力-磁对齐 + 连续 5 循环(Fig2/Fig3 数据)
2. `fig3c_BF_calibration` — ΔB–F 加载支轨迹 + 平台散点线性拟合(诚实标注 pooled R²)
3. `fig2f_response_event` — 高速率敲击事件 10–90% 上升/恢复标注 + 5 循环叠加 + 20 试次峰值 CV
4. `fig2f_cyclic_stability` — 5×5 阵列疲劳检查点内逐循环峰峰值稳定性

已有:`fig2e_geometry_sensitivity`(9 几何热图)、`fig2f_dynamics_durability`
(15 万次疲劳+响应时间分布)、`fig5d_pulse_demo`(脉搏)、`edx_wind_preliminary`。

## 三之二、2026-07-18 收到的示意图资产(`paper/figures/assets/`)

来源:`~/Desktop/机械臂+风洞论文图/`。

| 资产 | 去向 | 备注 |
|---|---|---|
| fig2b_fabrication_workflow.jpg (2243×1279) | **Fig. 2b,已接入拼版** | 原图两处英文拼写错误:"Vaccum"→Vacuum、"Demode"→Demould,矢量重绘时修正 |
| fig1b_hand_exploded_render.png (原 26750×15148,入库缩至 4000px) | Fig. 1b / 4a | 超高清 CAD 渲染 |
| fig5a_wind_tunnel_schematic.png (7070×4456) | Fig. 5a | 含 anemometer 的诚实风洞示意 |
| fig5d_wrist_patch_schematic.jpg | Fig. 5d | 腕部佩戴+皮肤剖面 |
| fig3f_bubble_touch_schematic.jpg | Fig. 3f | 指尖阵列钢球/气泡对比示意 |
| concept_toc_graphic.jpg | TOC/概念 | 疑似 AI 生成,受 Nature 政策限制 |
| inset1/2/3.png (仅 201px) | ⚠️ 太小不能印刷 | patch 实拍缩略图,**需原始高清照片** |

## 三之三、2026-07-18 高清 SEM 原图(`paper/figures/sem_hires/`,20 张)

来源:`~/Downloads/图 导出/`(原图 11378×7712 最高,共 28 MB,入库为
2400px JPEG 工作版;印刷时回原目录取原图)。**全部带真实 scale bar**,
解决了此前 c 面板无比例尺的问题。命名 = 直径-长度[-变体]-序号
(05-3=d0.5/L3,08-8-1=d0.8/L8 等;101=SEM 台上阵列全景 400 µm bar;
05-05-5 系列=颗粒分布特写 10 µm bar → Fig. 2a 素材)。
已用于拼版:c1=101(400 µm)、c2=08-8-1-01(300 µm)、
c3=05-3-04(100 µm bar + 402.5 µm 宽度标注,名义 d=0.5 mm → 实测收缩 ~19%,
可写进几何测量小节)。

⚠️ **政策提醒**:hand_array/airflow/pulse/bubble 若为 AI 工具生成,同样受
Springer Nature 禁令约束;若为设计师手绘/真实 CAD 渲染则可用——投稿前需向
期刊申明制作方式。建议保留源文件(.ai/.c4d/.blend 等)作为人工制作证据。

## 四、关键矛盾记录(需用户/实验方确认)

1. **Ecoflex vs PDMS**:真实流程图写 Ecoflex,可靠性样品目录是 PDMS_NdFeB,
   用户口头说 PDMS 10:1。当前处理:皮肤 patch=Ecoflex、几何/可靠性样品=PDMS,
   Methods 留确认标记。
2. **预取向存在**:真实流程图明确有"Pre-alignment under magnetic field"步骤,
   之前"就是一项"的理解不成立,Methods 已恢复两步(预取向+饱和充磁)。
3. **0.6 s vs 25 ms**:两个"响应时间"来自不同实验(手动加载沿 vs 敲击),
   论文只能用高速率敲击数据作为响应时间 claim。
