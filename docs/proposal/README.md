# 开题报告（hithesis 模板）

## 模板来源与版本

- 上游：[hithesis/hithesis](https://github.com/hithesis/hithesis)（经 gh-proxy.com 镜像克隆）
- 模板类：`hithesisart`（开题/中期报告专用类），`type=master, stage=opening, campus=harbin`
- 已生成：`hithesisart.cls` / `hithesisart.cfg` / `hithesis.bst`（由 `hithesis.ins` + `hithesis.dtx` 生成）
- 模板要求硕士生开题使用 `type=master`，深圳/威海/本部通过 `campus` 切换

## 开题报告结构（hithesisart, stage=opening）

以示例 `report_shenzhen_doctor_opening.tex` 为准，硕士开题同样适用：

| 章节 | 内容 |
|---|---|
| 封面 | `\hitsetup` 填：题目/学位类型/学院/学科/作者/学号/班级/导师 |
| §1 课题来源及研究的目的和意义 | 来源或研究背景；研究的目的及意义 |
| §2 国内外在该方向的研究现状及分析 | 国外研究现状；国内研究现状；国内外文献综述的简析 |
| §3 前期的理论研究与试验论证工作的结果 | 已完成的实验与证据链 |
| §4 学位论文的主要研究内容、实施方案及其可行性论证 | 主要研究内容（将来时态，不能只列目录）；实施方案及可行性论证 |
| §5 论文进度安排，预期达到的目标 | 进度安排；预期达到的目标 |
| §6 学位论文预期创新点 | 实证/方法/评估三个层面 |
| §7 为完成课题已具备和所需的条件、外协计划及经费 | 已具备与所需条件 |
| §8 预计研究过程中可能遇到的困难、问题，以及解决的途径 | 困难与对策 |
| §9 主要参考文献 | `hithesis.bst` 样式，BibTeX |

## 编译

```bash
cd docs/proposal
latexmk -xelatex report.tex    # 需要 MiKTeX/TeX Live + xelatex
```

`report.pdf` 为编译产物（11 页，含正文与参考文献）。封面占位信息（学院/学科/姓名/学号/导师）需按实际填写。

## 文件清单

- `report.tex` — 主文档（type=master, stage=opening, campus=harbin）
- `front/coverart.tex` — 封面 `\hitsetup` 配置
- `body/report_harbin_master_opening.tex` — 开题报告正文（§1–§9）
- `reference.bib` — 参考文献库（18 条，含 FPMC/PRME/ST-RNN/ARNN/LLM4Rec/FuseLLM/L2D/RouteLLM/距离衰减/重力模型/dip test/温度缩放/RETAIL 等）
- `hithesisart.cls` / `hithesisart.cfg` / `hithesis.bst` — 模板类与样式（由 dtx 生成）
- `hithesis.dtx` / `hithesis.ins` — 模板源，可重新生成 cls

## 待办（开题前）

- [ ] 封面信息按实际填写（学院/学科/姓名/学号/班级/导师）
- [ ] 国内研究现状补充 2–3 篇知网中文核心文献
- [ ] 确认答辩校区（campus=harbin/shenzhen/weihai）
