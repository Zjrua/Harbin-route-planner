# TravelPlanner 数据集（外部数据源）

来源：[osunlp/TravelPlanner](https://huggingface.co/datasets/osunlp/TravelPlanner)（OSU NLP，NeurIPS 2024）
协议：CC-BY-4.0
下载：通过 hf-mirror.com 镜像（huggingface.co 主站国内被墙）

## 数据规模
- test.csv：1000 条 query（含参考景点信息）
- validation.csv：181 条
- train.csv：46 条
- *_ref_info.jsonl：对应的详细参考信息（景点/餐厅/住宿，含坐标/地址/评分）

## 字段结构（test.csv）
| 字段 | 说明 |
|------|------|
| org | 起点（美国城市） |
| dest | 目的地（美国 127 个州/城市） |
| days | 天数（3/5/7） |
| date | 日期 |
| query | 自然语言需求（含预算、餐饮、交通等约束） |
| level | 难度（easy/medium/hard） |
| reference_information | 参考景点/餐厅/住宿信息（DataFrame 序列化） |

## 与本项目的关系（重要）
**TravelPlanner 是美国城市数据，与哈尔滨 POI 空间不重叠，不能直接并入训练。**

其价值在于：
1. **论文 Related Work**：作为旅游路线规划领域的权威 benchmark 引用，
   提升论文学术站位（NeurIPS 2024，被引 197+）
2. **方法论借鉴**：其多约束评估框架（预算满足、必需景点覆盖、餐饮/住宿规则）
   比本项目当前的 composite_score 更严谨，可启发评估维度设计
3. **约束设计参考**：query 中的预算/交通方式/饮食偏好等约束维度，
   可指导本项目 composite_score 的权重设计

## 后续可能用途（future work）
- 若做迁移学习/预训练，可作为预训练数据（但 POI 空间不同，收益不确定）
- 若扩展为 LLM agent 评估，可复用其 query 模板
