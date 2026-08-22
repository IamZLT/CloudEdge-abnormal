# 云边协同算法：代价–风险路由（CRR）

## 代码位置（可插拔）

| 路径 | 作用 |
| --- | --- |
| `src/collab_routing/base.py` | 接口：`RouteSignal` / `CloudState` / `RouteVerdict` / `CollabRouter` |
| `src/collab_routing/baseline.py` | 旧版 margin 启发式（对比基线） |
| `src/collab_routing/cost_risk.py` | **CRR 实现（默认）** |
| `src/collab_routing/registry.py` | `build_router` / `configure_routing` / `get_router` |
| `src/collab_routing/adapters.py` | 与 `RouteContext` / `EdgeFleet` 适配 |

切换算法（对比实验）：

```yaml
# configs/default.yaml → collab
route_policy: cost_risk   # 或 baseline
```

```python
from src.collab_routing import get_router, build_router

get_router(policy="baseline").decide(signal, cloud)
get_router(policy="cost_risk").decide(signal, cloud)
```

冒烟：`python scripts/smoke_collab_routing.py`

---

本文档只定义**一套**默认算法（CRR），直接建立在当前系统之上：

- 边侧 AD：分数 `s`、阈值 `τ`、裕度 `m = |s - τ|`、gallery 规模 `n_g`
- RouteAgent：读 CONTEXT，决定是否上云（现有 `rules_snap` 机制保留）
- 物理链路：`network_env` 已提供 RTT / 带宽 / 丢包 / outage
- 多节点：`edge_fleet` 多边缘共享同一云端

**结论：在现有边缘基础上，采用代价–风险路由（Cost–Risk Routing, CRR）作为参谋特征。**

默认部署（`enforce_context_rules: false`）：

1. 先算 CRR；写入 RouteAgent 的 CONTEXT **去重精简**：`edge_decision` + `edge_uncertainty` + `link_tier` + CRR 三件套（`suggest_upload` / `U` / `reason`），不再同时塞原始分数表与 `u_unc/c_net/c_cloud`
2. **RouteAgent LLM 主决策上云**：只输出 `upload` + 路由 `reason`；OK/NG 始终由边缘 AD / 云端 VLM 给出
3. **级联**：`edge_uncertainty=low` 时可跳过 LLM、直接用 CRR（`cascade_skip_low_uncertainty`）；mid/high/冷启动调用 LLM
4. 物理层 `try_upload` 失败 → `LOCAL_NET_FALLBACK`；JSON 解析失败 → 重试后保守本地

对比实验仍可用 `route_policy: baseline|cost_risk` 纯规则路径，或把 `enforce_context_rules: true` 恢复旧 snap。

不另开并行方案；场景复杂度、联邦 gallery、学到的路由器等不在本文范围。

---

## 1. 为什么用这一套

当前边侧已经具备决策所需的全部信号：

| 已有信号 | 现状用法 | CRR 用法 |
| --- | --- | --- |
| 裕度 `m = abs(s - τ)` | 与固定 `hard_margin` 比大小 | 转成不确定度（风险） |
| `n_g` | 仅冷启动强制上云 | 冷启动加成 + 调节有效裕度 |
| RTT / BW / loss / outage | 只影响 `try_upload` 成败 | **进入是否上云的决策** |
| 多节点同时上云 | 各自独立 | 云端有限并发下按效用准入 |

因此不需要新传感器、不需要第二套视觉编码器，只改路由规则与舰队准入。

---

## 2. 算法名称与目标

- **名称**：代价–风险路由（CRR）
- **单样本决策**：是否上传到云端 VLM
- **多样本 / 多节点决策**：云端并发有限时，接受哪些上传请求

隐式目标：在检测质量不明显下降的前提下，降低无效上云（弱网硬传、高置信仍上传、云端拥塞互抢）。

---

## 3. 输入 / 输出

### 3.1 单节点输入（CONTEXT + 链路快照）

- `edge_score`（s）、`edge_thr`（τ）、`edge_decision`
- `n_gallery`（n_g）
- 基准裕度 `h0`（配置项，对应现有 `hard_margin` / `thr_margin`）
- 链路 L：`rtt_ms`、`bandwidth_mbps`、`loss_prob`、`outage` / `profile`

### 3.2 舰队侧输入

- 各节点候选请求的效用 `U_i`
- 云端最大并发 `K`（配置 `cloud_admission.max_inflight`）
- 节点近期上云次数（公平性）

### 3.3 输出

- 单节点：`upload`（true / false），附带 `U` 与分解项（便于日志 / UI）
- 舰队：本窗接受集合；落选请求本地结案

---

## 4. CRR 计算步骤

### 步骤 1：硬门控

```text
if outage or network_profile == outage:
    return upload = false
```

与现网一致：断网不上云、不调云端。

### 步骤 2：有效裕度 `h_eff`

gallery 越大，本地越可信，不确定带收窄：

```text
h_eff = h0 * clip(n_ref / max(n_g, 1), 0.5, 2.0)
```

默认 `n_ref = 16`（与当前默认 gallery 规模对齐）。

### 步骤 3：不确定度（风险项）

```text
m = abs(s - τ)
u_unc = clip(1 - m / h_eff, 0, 1)
```

- `m` 远小于 `h_eff` → 靠近阈值 → `u_unc → 1`（该上云）
- `m` 远大于 `h_eff` → 本地很稳 → `u_unc → 0`（不上云）

### 步骤 4：链路代价（已有 network 字段直接算）

```text
c_net = clip(
    0.4 * rtt / rtt_ref
  + 0.4 * bw_ref / max(bw, eps)
  + 0.2 * loss,
  0, 1
)
```

默认：`rtt_ref = 80 ms`，`bw_ref = 50 Mbps`。
outage 已在步骤 1 返回，不再进入本式。

### 步骤 5：云端负载代价

```text
c_cloud = clip(Q / K, 0, 1)
```

- `Q`：当前云端 inflight + 排队长度
- `K`：最大并发
- 单节点 Demo 若无舰队信息，取 `c_cloud = 0`

### 步骤 6：效用与判决

```text
U = u_unc - w_n * c_net - w_c * c_cloud
if n_g == 0:
    U += w_g

upload = (U > 0)
```

默认权重：`w_n = 0.8`，`w_c = 0.5`，`w_g = 0.6`。

弱网保护（仍属同一判决）：若 `c_net >= 0.75` 且 `n_g > 0` 且 `u_unc < 0.85`，则强制 `upload = false`（链路很差且并非极度不确定 → 本地）。

### 步骤 7：多节点云端准入（同一效用，全局截断）

各节点先算本地 `U_i`；仅 `U_i > 0` 的请求进入候选。云端每窗最多接纳 `K` 个：

```text
score_i = U_i / (eps + c_net_i) - gamma * recent_cloud_i
```

按 `score_i` 降序取前 `K` 个实际上云；落选节点本窗本地结案。

---

## 5. 伪代码（完整）

```text
Algorithm CRR_Decide(ctx, link, cloud_state, cfg):
  # 1. hard gate
  if link.outage or ctx.network_profile == "outage":
      return False, U = -inf

  # 2-3. risk
  m = abs(ctx.edge_score - ctx.edge_thr)
  h_eff = cfg.h0 * clip(cfg.n_ref / max(ctx.n_gallery, 1), 0.5, 2.0)
  u_unc = clip(1 - m / h_eff, 0, 1)

  # 4. link cost
  c_net = clip(
      0.4 * link.rtt_ms / cfg.rtt_ref
    + 0.4 * cfg.bw_ref / max(link.bandwidth_mbps, 1e-3)
    + 0.2 * link.loss_prob,
      0, 1)

  # 5. cloud load
  c_cloud = clip(cloud_state.queue_or_inflight / cfg.K, 0, 1)

  # 6. utility
  U = u_unc - cfg.w_n * c_net - cfg.w_c * c_cloud
  if ctx.n_gallery == 0:
      U += cfg.w_g

  if c_net >= 0.75 and ctx.n_gallery > 0 and u_unc < 0.85:
      return False, U

  return (U > 0), U


Algorithm CRR_Admit(candidates, K, gamma):
  C = [c for c in candidates if c.U > 0]
  for c in C:
      c.score = c.U / (1e-6 + c.c_net) - gamma * c.recent_cloud
  sort C by score desc
  return C[:K]   # accept; others stay local this window
```

---

## 6. 接到现有代码的位置

| 组件 | 改动 |
| --- | --- |
| `heuristic_upload()` | 改为 CRR 步骤 1–6（单节点；`c_cloud=0` 或读舰队状态） |
| RouteAgent `rules_snap` | snap 目标改为 CRR 结果（不再用旧 margin 规则） |
| RouteAgent prompt 规则条文 | 与 CRR 文字对齐（outage / 冷启动 / 不确定且链路可承受才上云） |
| `edge_fleet` | 维护 inflight、recent_cloud；提供 `CRR_Admit` |
| `try_upload` | 保持；只对准入成功的请求调用 |
| 配置 `configs/default.yaml` | 增加 `collab.cost_risk` 权重与参考尺度 |

不改边侧 AD 骨干，不改云端 VLM 结构。

---

## 7. 默认配置

```yaml
collab:
  hard_margin: 0.05          # h0
  route_policy: cost_risk    # 启用 CRR
  cost_risk:
    n_ref: 16
    w_n: 0.8
    w_c: 0.5
    w_g: 0.6
    rtt_ref_ms: 80
    bw_ref_mbps: 50
    weak_c_net: 0.75
    force_unc: 0.85
  cloud_admission:
    max_inflight: 2
    fairness_gamma: 0.1
```

---

## 8. 行为直觉（对照现状）

| 情形 | 旧启发式 | CRR |
| --- | --- | --- |
| 高置信（m 大）+ 好链路 | 不上云 | 不上云 |
| 近阈值 + 好链路 | 上云 | 上云（U > 0） |
| 近阈值 + 很差链路 | 仍尝试上云，常 fallback | 倾向本地（代价压过风险） |
| 冷启动 n_g = 0 | 上云 | 上云（w_g 加成），除非 outage |
| 多节点同时抢云 | 无协调 | 按 U / c_net 取 Top-K |

---

## 9. 验收

固定边侧 AD 与云端权重，只替换路由：

1. 正常 geo 链路：F1 不低于旧启发式；上云率下降或持平
2. 弱网 / 高拥塞：`LOCAL_NET_FALLBACK` 减少，端到端更稳
3. 3 节点同时高峰：云端排队可控，高不确定且链路尚可的样本优先上云

---

## 10. 一句话

在现有边缘 AD + 链路快照 + RouteAgent 上，用代价–风险效用

```text
U = u_unc - w_n * c_net - w_c * c_cloud (+ 冷启动加成)
```

决定上云，并用同一效用做多节点云端 Top-K 准入。
