# ItineraryTransformer 项目技术诊断报告（改名前为 RouteTransformer）

> 生成时间：2026-07-26  
> 范围：代码、配置、checkpoint、训练日志、消融数据、论文一致性  
> 目的：竞赛结束后复盘，为后续技术方向决策提供事实依据。**只诊断，不改代码。**

---

## TL;DR（三句话结论）

1. **过拟合是结构性的，不是调参问题**：4.57M 参数 / 4134 训练样本 ≈ 每参数 0.9 样本，train/val gap=2.1 是数学必然，`--resume` 继续训练只会更糟。
2. **"三大创新"里只有 Engram 有微弱证据，MHC 反向，Muon 零实测**：Mu作为"核心创新"被大篇幅介绍，但实际主训练和全部消融都用 AdamW——这是论文最大风险点。
3. **更根本的隐患是数据循环论证**：5168 条样本里 5000 条是规则合成，而评估指标又由同一套规则生成，composite_score=0.88 几乎不能说明问题。

---

## 一、远程仓库状态

| 分支 | 与 main 的关系 | 内容 |
|---|---|---|
| `main` | — | 当前工作分支，与 `origin/main` 完全同步（0/0），工作区干净 |
| `origin/mps-training` | +6 commits | 全部是论文/文档提交（导师意见修复、LaTeX→Word、参赛资料、MPS 适配），**与训练代码无关**，不影响 Windows/RTX 4090 训练 |

**结论：无需 merge 任何远程分支。代码基线是干净的。**

---

## 二、核心问题诊断

### 问题 1：过拟合是结构性的，无法靠"继续训练"解决 ⚠️

**事实（来自 `output/training_loss.csv`）：**

```
epoch    train_loss    val_loss
1        9.0631        8.7592
50       4.5167        5.3464
106      3.0382        4.9041   ← best (早停基准)
121      2.7903        4.9474   ← 早停触发
```

- **train/val gap = 2.79 → 4.90 差 2.11**，且 val 在 epoch 106 后单调反弹 15 轮
- 模型参数：4,569,976
- 训练样本：4,134（routes.npy 实测 5168 条，0.8 划分）
- **参数/样本比 ≈ 1106:1**（严重失衡，NLP 大模型这个比例通常 < 10:1 才健康）

**根因**：不是 dropout（已 0.2）、不是 lr、不是优化器——是**模型容量相对数据量太大**。Encoder-Decoder Transformer 对 4134 个样本而言是杀鸡用牛刀。

**推论**：直接 `python -m src.train --resume checkpoints/best_model.pt` 继续跑到 epoch 200，val_loss 不会改善（早停机制正是为此设计）。**"继续训练"这个动作本身在当前配置下无意义。**

---

### 问题 2：Muon 作为"核心创新"零实测，且主训练用的是 AdamW 🔴

这是本报告**最严重**的发现。

#### 2.1 主训练实际用 AdamW（checkpoint 实锤）

`checkpoints/best_model.pt` 的 optimizer_state_dict 第一个参数的 state 键为：

```
['step', 'exp_avg', 'exp_avg_sq']
param_groups[0] keys: ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'decoupled_weight_decay', ...]
```

- `exp_avg` / `exp_avg_sq` / `betas` / `decoupled_weight_decay` 是 **AdamW 的特征性键**
- Muon 只有 `momentum_buffer`，没有 `exp_avg_sq`

**结论：val_loss=4.9041 的最佳模型 100% 用 AdamW 训练。**

#### 2.2 git 历史显示配置改写发生在"训练之后"

commit `4b32168`（2026-05-08 "POI规模扩展至10K"）在同一次提交里做了两件矛盾的事：

```diff
 optimizer:
-  name: "adamw"  # muon / adam / adamw
+  name: "muon"   # muon / adam / adamw
```

同时 commit message 写："训练结果：epoch 106 best val_loss=4.9041"。

**也就是说：训练在配置还是 `adamw` 时完成，改成 `muon` 是训练结束后、写论文那一刻才做的。** CLAUDE.md 第 241 行那句"AdamW"反而是对的，default.yaml 第 52 行的 `muon` 反而是错的。

#### 2.3 消融实验也全部用 AdamW

`scripts/run_ablation.py:57` 硬编码：

```python
optimizer = torch.optim.AdamW(grouped, weight_decay=opt_cfg["weight_decay"])
```

**完全忽略 config 里的 `optimizer.name`**。7 组消融（full / no_engram / no_mhc / no_engram_mhc / baseline / k3 / k10）无一例外都是 AdamW。

#### 2.4 论文怎么写的？（关键）

论文**没有造假**，反而很诚实：

| 论文位置 | 原文 |
|---|---|
| 第 149 行 | "采用AdamW优化器（batch_size=256, AMP混合精度）" |
| 第 985 行 | "所有消融实验统一采用AdamW优化器（**而非Muon**）" |
| 第 996-1002 行 | 表 5 每行都标注 "AdamW" |

**但矛盾在于**：

| 论文位置 | 原文 | 问题 |
|---|---|---|
| 第 147 行（摘要） | "融合三项核心创新...（3）**Muon正交化优化器**" | 把 Muon 列为三大创新之一 |
| 第 213 行（引言） | "（3）Muon矩阵正交化优化器：替代Adam..." | 详述其优势 |
| 第 636-667 行 | **整整一节**专门讲 Muon，含公式、Newton-Schulz 迭代、Kimi 缩放 | 大篇幅技术介绍 |
| 第 1134 行（结论） | "融合...Muon梯度正交化" | 再次强调 |

**风险定性**：不是学术造假（论文如实写了用 AdamW），但属于**严重的过度宣称（overclaim）**——用一个从未在自己的实验里跑过、从未与 AdamW 做过对比的东西，作为三大核心创新之一，并占据论文显著篇幅。答辩老师如果问"Muon 比 AdamW 好多少？请展示对比数据"，**没有任何一条数据可以回答**。

`configs/ablation.yaml` 里那个 `adamw` block 也是死代码：注释自己写"当前默认已是AdamW"，且 `run_ablation.py` 的实验列表（217-225 行）根本没包含它。

---

### 问题 3：MHC 在当前数据规模下净负贡献 🟡

**事实（来自 `output/ablation_results.csv`）：**

| 变体 | composite | 平均距离 km | 相对 Full |
|---|---|---|---|
| Full (K=5, 含 MHC) | 0.8747 | 32.3 | — |
| **-MHC**（去掉双曲约束） | **0.8753** | **27.2** | **+0.0006（更好）** |
| -Engram-MHC | 0.8735 | 34.5 | -0.0012 |
| Baseline（都去掉） | 0.8676 | 63.0 | -0.0071 |

去掉 MHC 后：composite 略升，路线距离从 32.3km 降到 27.2km（-16%）。

论文第 1098 行自己也承认："MHC双曲约束在当前数据规模下贡献有限... 变化均在统计波动范围内"。

**定性**：MHC 只占 0.3% 参数（12K），数学复杂度高（庞加莱球、expmap/logmap/geodesic），维护成本不低，但在当前规模下**没有任何正向贡献**。它可能在大规模、强层级结构的数据上才有价值（这是它原始论文 Nickel & Kiela 2017 的设定），但在 10000 POI + 合成路线这个场景下不成立。

---

### 问题 4：数据循环论证（最根本的隐患）🔴

这是比过拟合更深层的问题，且**论文里没有讨论**。

#### 4.1 训练数据的构成

```
真实数据：168 条 XHS 路线（3.2%）
合成数据：5000 条（96.8%）  ← scripts/prepare_data.py（原 prepare_10k_data.py） 用规则生成
─────────────────────────────────
总计：5168 条
```

合成规则（CLAUDE.md:91-94）：类别感知近邻随机游走（80%景点/15%购物/5%餐饮）+ 距离^-2 × 评分加权 + 餐饮住宿规律插入。

#### 4.2 评估指标也由相似规则生成

`scripts/run_ablation.py` 的 `evaluate_route()` 用以下方式生成测试路线并打分：

```python
# 生成时：约束 Beam Search + 活动类型约束 + 步行集群 masking
# 评估时：composite = 0.3*distance + 0.25*time + 0.25*satisfaction + 0.2*diversity
```

距离、满意度（评分）、多样性这些指标**和合成路线时用的"距离^-2 × 评分"加权高度同源**。

#### 4.3 后果

模型在学合成器的规则 → 你用同样规则生成的 test set 评估 → composite_score 高。

**这个 0.88 的数字主要衡量"模型多大程度复现了你的合成器"，而不是"模型多大程度学到了真实游客的偏好"。** 论文核心实验（表 7）的所有对比都建立在这个循环之上。

真正能说明问题的是 168 条真实 XHS 路线上的表现，但：
- 它们在训练集里只占 3.2%，被合成数据淹没
- 评估时（`evaluate_route` 第 178-184 行）用的是"高评分景点作为起点、模型 beam search 生成"，**根本不是在 XHS 真实路线上对比**——是模型自生成 vs 规则评分

#### 4.4 routes.npy 的工程问题（附带）

`routes.npy` 是 `dtype=object` 的变长数组（实测 5168 条，长度 min/median/max = 2/14/33）。`np.load(..., allow_pickle=False)` 会直接报错。这本身不算 bug（dataset.py 能正确读），但说明数据管线没有一个统一的 padding/对齐约定，调试和扩展时容易踩坑。

---

### 问题 5：Engram 的实现与论文描述有出入 🟡

#### 5.1 检索阶段没有真正用上门控

`engram.py` 里 `forward()`（第 95-114 行）实现了完整的"门控融合"（query + sigmoid(gate) * context），但 `transformer.py:140-142` 调用时只调用了 `retrieve()`：

```python
query = encoder_output.mean(dim=1)
retrieved, _ = self.engram.retrieve(query)
engram_memory = retrieved
```

**`forward()` 这个带门控融合的方法在整个代码库里从未被调用**（grep 确认）。也就是说论文描述的"可学习门控融合"机制实际没进入训练。`gate` 参数和 `out_proj` 虽然在 `__init__` 里定义了，但梯度永远流不过去——它们是死参数。

#### 5.2 memory_values == memory_keys

`build_memory()` 第 64-65 行：

```python
self.memory_keys[:n] = keys
self.memory_values[:n] = keys.clone()  # ← keys 和 values 完全相同
```

keys 和 values 是同一份数据（POI embedding 的平均池化）。这让 Engram 退化成"基于自身表征的自检索"，失去了"键值解耦"的意义（标准记忆网络 keys 用于寻址、values 存储不同模态信息）。

#### 5.3 季节权重是死参数

`season_weights`（第 57 行）是 `nn.Parameter`，但 `update_season()`（第 116 行）从未在 train/inference 里被调用。这个参数虽然参与 autograd，但没有梯度信号——也是死参数。

**小结**：Engram 模块的实际行为 = "对 encoder 输出做一次余弦相似度 top-k 检索 + softmax 加权"，比论文描述的简洁得多。这不是 bug，但论文里关于"门控融合"、"季节感知"的描述与代码实现不符。

---

## 三、技术方向建议（供评估）

竞赛已结束，所以下面的建议面向"**如果这是一个要持续做下去的项目 / 要发论文 / 要写进简历**"，按推荐度排序。

### 方向 A：把 Muon 这个洞补上（最小改动，最高优先级）✅

**动机**：Mu作为创新点写进了摘要、引言、专节、结论，但零实测。这是最容易被攻击的点。

**做法**：用 `configs/default.yaml` 现成的 `muon` 配置（已经写好了），跑一次完整主训练 + 一组消融对比。代码层面 `src/optim/muon.py` 已经实现且通过测试（`tests/test_engram.py` 之外没看到 muon 测试，需补），`build_optimizer` 也已支持 muon 分支。

**预期工作量**：1 次主训练（121 epoch × 40s ≈ 1.5 小时）+ 1 组 AdamW vs Muon 对比消融。可能的结果：
- Muon > AdamW → 论文论点成立，补上数据即可
- Muon ≤ AdamW → 诚实写进论文，把 Muon 从"核心创新"降级为"探索性尝试"，反而加分

**风险**：如果 Muon 不如 AdamW，需要重写论文相关章节。但这是**必须知道的真相**。

---

### 方向 B：拆出真实 XHS 路线做独立评估（学术价值最高）✅✅

**动机**：打破数据循环论证。这是整个项目从"工程 demo"升级为"可发表研究"的关键一步。

**做法**：
1. 把 168 条真实 XHS 路线从训练集**完全剥离**，作为 hold-out test set
2. 用 5000 条合成数据训练（或再加部分增强）
3. 在 168 条真实路线上评估：next-POI 预测准确率、路线长度分布对比、人工抽样检查合理性
4. 同时跑一个简单的基线（最热门 POI / 距离最近邻）作为参照

**预期收获**：哪怕模型在真实路线上表现一般，"我们诚实地报告了合成→真实的泛化差距"本身就是高质量的科研贡献，比 composite=0.88 有说服力得多。

**风险**：168 条样本可能太少，统计意义有限。但即便如此，也胜过当前的循环评估。

---

### 方向 C：加一个 OR/启发式基线对比（质疑 Transformer 必要性）🤔

**动机**：路线规划本质是 TSP/VRP 变体，是运筹学的传统领地。用 Transformer 做 seq2seq 需要论证"为什么学习比启发式好"。目前论文缺少这个对比。

**做法**：
- **轻量基线**：最近邻 + 2-opt（`src/inference.py` 里其实已经有 `optimize_route_order` 用了 2-opt，但只用作后处理，没作为独立基线评估）
- **重量基线**：Google OR-Tools 的 VRP solver，给定同样的 POI、距离矩阵、约束

**预期结果**：OR 基线在"距离最优"上几乎一定赢 Transformer。但 Transformer 可能在"符合人类偏好"（多样性、节奏感）上有优势——**这才是你真正的卖点**，需要专门设计评估来体现。

**风险**：如果 OR 基线全面碾压，等于否定了项目的核心价值。但**知道真相比自我感觉良好重要**。

---

### 方向 D：精简模型 + 缩容量，对症下药治过拟合 🟡

**动机**：当前过拟合是结构性的。

**做法**（按收益排序）：
1. **砍模型容量**：d_model 128→64，encoder/decoder 3→2 层。参数从 4.57M 降到 ~1.5M，更匹配 4134 样本
2. **数据增强**：路线的随机扰动、子路线采样、反向路线（如果方向对称）
3. **更强正则**：dropout 0.2→0.4，weight_decay 1e-4→1e-3，加 label smoothing
4. **K=3 替代 K=5**：消融显示 K=3 是全局最优（composite 0.8802）

**单独做这个方向价值有限**——治标（过拟合）不治本（数据循环）。建议作为方向 B 的配套。

---

### 方向 E：转向 RL 微调（CLAUDE.md 里提到的 roadmap）🔴 不推荐

CLAUDE.md 第 200-205 行提到 Self-Critical Seq2Seq / DPO 微调。

**不推荐的原因**：
- RL 微调需要可靠的 reward signal。当前 reward 就是 composite_score，而它本身有循环论证问题（见问题 4）
- RL 在 4134 样本 + 4.57M 参数的过拟合基础上，会进一步过拟合 reward
- 工程复杂度高，收益不确定
- **应该先解决问题 B（真实评估），再考虑 RL**，否则是在沙地上盖楼

---

## 四、论文一致性核查清单

| 检查项 | 论文说法 | 实际情况 | 状态 |
|---|---|---|---|
| 主训练优化器 | 第149行："AdamW" | checkpoint 实测 AdamW | ✅ 一致 |
| 消融优化器 | 第985行："统一AdamW而非Muon" | run_ablation.py:57 硬编码 AdamW | ✅ 一致 |
| Muon 是否实测 | 摘要/引言/专节/结论都作为创新 | 主训练和消融都没用 | 🔴 过度宣称 |
| MHC 贡献 | 第1098行："统计波动范围内" | 消融显示去掉反而更好 | 🟡 偏乐观但可接受 |
| Engram 门控融合 | 论文描述可学习门控 | engram.forward() 从未被调用 | 🟡 描述>实现 |
| Engram 季节感知 | 未在论文详述 | season_weights 死参数 | 🟡 代码有但未用 |
| 数据规模 | 5168条（168 XHS + 5000合成） | routes.npy 实测 5168 条 | ✅ 一致 |
| 模型参数量 | 4,569,976 | — | ✅（按 CLAUDE.md） |
| 最佳 val_loss | 4.9041 @ epoch 106 | checkpoint epoch=105(0-indexed), loss=4.9041 | ✅ 一致 |

---

## 五、立即可做的低风险修复（不需要重新训练）

这些不改变模型行为，只修正文档与代码的一致性：

1. **`default.yaml:52`**：`name: "muon"` 改回 `name: "adamw"`，与实际训练配置一致（或保留 muon 但加注释说明这是"待验证配置"）
2. **`CLAUDE.md:241`**：保持 "AdamW"（已正确），但补一句说明 default.yaml 的 muon 是未实测的预期配置
3. **`engram.py`**：要么调用 `forward()` 让门控真正参与训练，要么删掉 `gate`/`out_proj`/`season_weights` 这些死参数，避免误导
4. **`scripts/run_ablation.sh`**：这是过时脚本（引用了不存在的 experiment key 和 curvature 变体），应删除或标注 deprecated
5. **`configs/ablation.yaml` 里的 `adamw` block**：死配置，删除或注明

---

## 六、决策矩阵

| 方向 | 工作量 | 学术价值 | 风险 | 推荐场景 |
|---|---|---|---|---|
| A. 补 Muon 实验 | 低（几小时） | 中（堵住最大漏洞） | 中（可能要改论文） | **必做** |
| B. 真实数据独立评估 | 中（1-2天） | **高**（核心贡献升级） | 低 | **强烈推荐** |
| C. OR 基线对比 | 中（1-2天） | 高（质疑必要性） | 高（可能否定项目） | 想发论文/写简历 |
| D. 精简模型治过拟合 | 低（半天） | 低（治标） | 低 | 作为 B 的配套 |
| E. RL 微调 | 高（1周+） | 不确定 | 高（reward 循环） | **不推荐**（先做 B） |

---

## 附：本报告的所有一手证据来源

| 论断 | 证据文件 |
|---|---|
| 主训练用 AdamW | `checkpoints/best_model.pt` 的 optimizer state（exp_avg/exp_avg_sq） |
| 配置改写时间点 | `git log -p 4b32168 -- configs/default.yaml` |
| 消融硬编码 AdamW | `scripts/run_ablation.py:57` |
| 过拟合 gap=2.1 | `output/training_loss.csv` 第 106/121 行 |
| 数据循环论证 | `scripts/prepare_data.py（原 prepare_10k_data.py）`（合成规则）+ `run_ablation.py:152-207`（评估规则） |
| MHC 净负贡献 | `output/ablation_results.csv` 第 2-4 行 |
| Engram 门控未启用 | `src/models/transformer.py:140-142`（只调 retrieve 不调 forward） |
| routes.npy 是 object array | `np.load` 实测 dtype=object，shape=(5168,) |
| 论文 Muon 篇幅 | `paper/main.tex` 行 147/213/636-667/1134 |
| 论文已诚实写 AdamW | `paper/main.tex` 行 149/985/996-1002 |

