# 数据采集 SOP 总集(按优先级排序)

版本 2026-07-18。每个 SOP 执行前先读对应 verdict 文档
(`wind_data_verdict.md`, `pulse_data_verdict.md`)吸取已踩过的坑:
漂移、EMI 混杂、装置中途被碰、无文件内基线、无对照组。

## 通用规范(所有 SOP 共用)

**数据格式**
- 每 trial 一个 CSV + 一个同名 JSON metadata;绝不把条件编码在文件名里当唯一记录。
- CSV 列:`t_ms, s1_Bx, s1_By, s1_Bz, ..., [Fx, Fy, Fz], [probe_x, probe_y, probe_z], [T_degC]`,单位 µT / mN / mm / °C,写进表头。
- JSON 必含:date, operator, patch_id, cilia geometry (L/d/pitch), sensor config
  (gain/OSR/filter/ODR), 条件参数, trial index, 随机化种子, 异常备注。
- 记录**实测**采样率(时间戳推算),不是标称值。

**防混杂三原则**
1. 任何会动的铁磁/电机部件(风扇、机械臂电机、线缆)先做**空载 sham**:
   装置运行但不接触/不吹到样品,记录 ≥60 s,量化 EMI 底。
2. 每 trial 内含 ≥1 s 无刺激基线(文件内基线,抵消漂移)。
3. 装置一旦被碰立即在 JSON 里记 `disturbed: true`,该 trial 后重录基线。

**统计规范**:n = 独立器件/受试者;技术重复单独记;条件顺序随机化并保存顺序表。

---

## SOP-1(最优先)Fig. 3:2×2 阵列 + Nano17 力标定 + 未见位置反演

**硬件准备**
1. 2×2 MLX90393 PCB:记录 4 颗芯片真实中心坐标(CAD 值 + 显微镜复核)、
   sensor pitch、patch 尺寸;贴纤毛层(当前默认 L5/d0.8,Fig. 2 出结果后换最优几何)。
2. xArm 末端装 Nano17,Nano17 前端装**非磁探针**(PEEK 或尼龙,推荐 2 mm 半球头;
   陶瓷备选)。禁止不锈钢探针、钢制紧固件在探针 30 mm 内。
3. 样品刚性固定在非磁基板(亚克力/铝),水平校准(探针法向偏差 < 2°)。

**上机前检查(全部通过才开始)**
- C1 探针磁性检查:探针距皮肤 1 mm 悬停 30 s 不接触,|ΔB| < 3σ 噪声。
- C2 机械臂 EMI 检查:臂全速在样品上方 5 cm 做典型轨迹,记录 |ΔB|;若 > 3σ,
  增大 standoff 或换轨迹速度,并把该值记入 Methods。
- C3 Nano17 零漂:空载 5 min,漂移 < 1 mN。
- C4 同步验证:探针轻敲样品 10 次,MLX 信号峰与 Nano17 力峰时差 < 10 ms。

**同步方案**:单主机采集,MLX(≥200 Hz)与 Nano17(≥1 kHz)各自打主机时间戳;
每 trial 开头由 xArm 触发一次"敲击同步脉冲"(接触-离开 <100 ms)作硬对齐锚点。

**Fig. 3c 指纹采集**(先做,约半天)
- 中心位置:Fn = 10/20/50/100 mN;剪切 Fs = 10/20/50/100 mN,
  θ = 0°…315° 每 45°;每条件 5 重复,顺序随机。
- 每 trial:1 s 基线 → 加载(0.5 mm/s)→ 1 s 保持 → 卸载 → 1 s 恢复。

**Fig. 3d 网格采集**(核心,约 2.5–3 h/session × 3 sessions)
- 7×7 = 49 位置覆盖整个 2×2 单元胞(位置间距 = sensor pitch/6 量级,按实际 pitch 定)。
- 每位置 Fn = 10/30/100 mN × 5 重复 = 735 trials/session。
- 位置与力顺序整体随机化;每 30 trial 回中心位置打一个"锚点 trial"监控漂移。
- 3 个 session 间重新装夹样品(测 cross-session 泛化)。
- 温度全程记录(MLX 自带温度通道)。

**分析划分(写死,防泄漏)**
- 按**位置**分:49 位置随机 60%/20%/20% = 29 train / 10 val / 10 unseen test;
  同一 trial 的任何时间帧不得跨集合;unseen test 位置只在最终评估用一次。
- 模型顺序:nearest-neighbour → ridge/poly → random forest → compact MLP;
  分三阶段输出 (x,y) → (x,y,Fn) → (x,y,Fn,Fx,Fy)。
- 验收线:定位 RMSE < sensor pitch(可称 local spatial resolution);
  < 0.5×pitch 为强;力 NRMSE < 15%;方向中位误差 < 15°。

## SOP-2 Fig. 2:几何矩阵 + 对照组(决定 cilia-enhanced 能否成立)

**样品矩阵**(每种 n = 3 独立 patch)
- 6 几何:L ∈ {3,5,8} mm × d ∈ {0.5,0.8} mm,pitch 以模具实测为准。
- 4 对照:flat magnetic film、short pillar(≤1 mm)、non-magnetic cilia、bare sensor。
- **关键控制**:flat film 与纤毛组的 NdFeB 总量/单位面积一致(称重记录),
  且记录每组空载 |B|;否则增强倍数会被磁料量差异顶掉——这是审稿第一问。

**测量清单(每 patch)**
1. 几何:≥30 根纤毛(3 patch 合计≥90),显微镜测 L/d/pitch,mean±s.d.。
2. 空载基线 60 s(σ_B 用于 LOD)。
3. 法向力 5 级(10/20/50/100/200 mN;若 5 mN 信噪可行再加)× 5 重复。
4. 剪切力 5 级 × 5 重复(用 SOP-1 平台)。
5. 阶跃响应 ×10:上升/恢复时间。
6. 频率扫描:0.1/0.5/1/2/5/10 Hz 正弦压载,各 30 s。
7. 耐久:10,000 次循环(50 mN, 2 Hz),每 1,000 次测一次灵敏度。
8. 漂移:30 min 静置基线(争取 2 h 一次)。

**输出与统计**:S = d|ΔB|/dF(线性段斜率),G = S_cilia/S_flat,LOD = 3σ_B/S;
组间双侧检验(ANOVA + Holm 校正),报 effect size 和确切 P;
G 的 95% CI 由 3 patch × bootstrap。只有 G 显著 > 1 才启用 "cilia-enhanced"。

## SOP-3 Fig. 4:44 节点整手

**Bring-up(先于一切实验)**
- 逐节点表(ED Fig. 4):baseline、noise σ、gain、安装朝向、MUX 通道、状态;
  坏道记 dead,不许悄悄插值。
- 实测完整 44 点 frame rate 与 10 min 丢帧率(这是论文数字,禁用芯片标称)。

**No-contact 伪影协议(比接触实验优先)**
- 7 动作:open hand / 单指屈 / fist / wrist pitch / wrist roll / 整手旋转 / 拉线缆,
  每动作 n = 10,每次 10 s。
- 同日做标准接触:palm 50 mN、fingertip 30 mN、thumb 30 mN(SOP-1 平台)。
- 加一颗**无纤毛 reference MLX90393**(手背或腕部),做 ΔB_corr = ΔB − α·ΔB_ref;
  报 CMR 前/后与 false-positive rate。

**Known-position 验证**:固定手模,30 真值位(palm 10 / fingertip 5 / pad 10 /
thumb 3 / edge 2)× 10/30/100 mN × 5 重复 = 450 trials;输出定位误差、
region 混淆矩阵、false-hotspot ratio。

**场景映射**:palm press / pinch / cylinder / ball / lateral shear / gentle contact,
每场景 n = 10,统一色标;力标定完成前只标 |ΔB| map。

## SOP-4 气流补测(修复 2025-07-17 数据缺陷)

1. **文件内 ON/OFF 循环**:每记录 = 30 s 关 / 30 s 开 × 5;ΔB 按周期配对计算。
2. **EMI 对照**:bare sensor(无纤毛)同位置全风速扫一遍;再做"风扇转、
   风道堵死"sham——两者合起来分离电机磁场与气动响应。
3. 风速点:0/0.5/1/2/3/4/5 m/s(低速段是弱气流主张所在,必须有);
   每速 5 独立 run,顺序随机,风速计逐 run 记录。
4. 风扇离轴布置或加风道,增大电机-传感器距离;中途禁碰装置(trial-9 教训)。
5. 输出:ΔB–v 拟合 + R²、上/下行迟滞、响应/恢复时间、重复性。

## SOP-5 脉搏 + ECG(修复现有数据无真值问题)

1. **伦理先行**:IRB 批准 + 知情同意,然后才碰受试者。
2. **硬同步**:ECG 与 MLX 共主机打时间戳,再加硬件触发(同一 GPIO 脉冲
   进两路数据流);验收:敲击测试时差 < 10 ms。建议再加 PPG。
3. 受试者:探索 n = 5 → 主实验 n = 10–20;每人 3 session,session 间重新贴附;
   记录 patch 位置、贴附压力、左右腕、姿势、subject ID。
4. 位置偏移图:centre / ±5 mm radial-ulnar / ±10 mm proximal-distal,各 3×60 s。
5. 输出:HR MAE(vs ECG)、ECG-to-pulse delay、beat 对齐波形、SNR、CV;
   dicrotic notch 仅在跨 beat 稳定时报告。

## 执行顺序建议

| 周 | 内容 |
|---|---|
| 1 | SOP-1 硬件搭建 + 检查清单 C1–C4 + Fig. 3c 指纹 |
| 2–3 | SOP-1 网格 3 sessions + 反演建模 |
| 3–4 | SOP-2 样品制备(6 几何 + 4 对照 × 3 patch) |
| 4–5 | SOP-2 全部表征 |
| 5–6 | SOP-3 bring-up + 伪影协议 |
| 6–7 | SOP-3 known-position + 场景 |
| 并行 | SOP-4(半天可完成)、SOP-5(待伦理批复) |
