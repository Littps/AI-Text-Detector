# 引擎目录

> 本页列出目录中的全部检测引擎、它们的真实体积与已知限制，以及**自动模式怎么选引擎**、**模型管理里的默认集与风险**。
> **"推荐"只表示默认勾选，不代表作者主力配置**；任何引擎都不锁死，可自由勾选。

## 分类

| 类型 | 含义 | 输出粒度 |
|---|---|---|
| `classifier` | 微调分类头（BERT / RoBERTa / DeBERTa / ModernBERT） | 逐句打分 |
| `method` | 零样本统计方法（困惑度、曲率、交叉困惑度） | 逐句或整篇 |
| `router` | 语言识别基础设施 | 不产出 AI 率 |

## 目录（12 个）

| 引擎 | 语言 | 类型 | 下载体积 | 驻留(fp16) | 许可 | 状态 |
|---|---|---|---|---|---|---|
| `simpleai-zh` | 中 | classifier | 0.41GB | ~202MB | Apache-2.0 | ✅ 已验收 |
| `xiaoyu-zh` | 中 | classifier | 0.41GB | ~202MB | Apache-2.0 | ✅ 已验收 |
| `aigc-zhv3` | 中 | classifier | 0.41GB | ~193MB | Apache-2.0 | ✅ 已验收（v3 新一代） |
| `aigc-zhv3short` | 中 | classifier | 0.41GB | ~193MB | Apache-2.0 | ✅ 已验收（短句特化） |
| `modernbert-raid-mage` | 英 | classifier | 0.60GB | ~300MB | Apache-2.0 | ✅ 已验收 |
| `desklib-en` | 英 | classifier | 1.75GB | ~828MB | MIT | ✅ 已验收（文档级） |
| `vanguard-en` | 英 | classifier | 1.59GB | ~755MB | MIT | ✅ 已验收 |
| `aigc-env3` | 英 | classifier | 0.50GB | ~250MB | Apache-2.0 | ✅ 已验收（v3 新一代） |
| `gltr` | 英 | method | 0.55GB | ~243MB | 方法实现 | ✅ 已验收 |
| `r1-lang-router` | 多 | router | 1.13GB | ~560MB | MIT | ✅ 已验收（基建） |
| `fast-detectgpt` | 英 | method | **10.67GB** | ~5.4GB | 方法实现 | ⚠️ 实现已接线，权重未下载，阈值未标定 |
| `binoculars` | 英 | method | **28.87GB** | **~28GB** | 实现 MIT / 权重按仓库 | ⚠️ 实现已落地，4bit 未实现，8GB 卡 OOM |

> 体积为 manifest 的真实字节数（fp32 原件）。`keep-original` 模式加载时现场转 fp16；`fp16-only` 模式可省约一半磁盘（见[架构设计](ARCHITECTURE.md)）。

## 自动模式：怎么选引擎

四个模式里三个是自动的（`fast` / `balanced` / `precise`），`custom` 手动勾选。自动模式在**前端**解析成具体引擎 id 列表再交给后端执行——不是后端猜的，也不做二次增删。

| 模式 | 中文文本 | 英文文本 |
|---|---|---|
| `fast` 高速（2 / 2） | `aigc-zhv3`、`simpleai-zh` | `modernbert-raid-mage`、`vanguard-en` |
| `balanced` 均衡（3 / 4） | `aigc-zhv3`、`simpleai-zh`、`xiaoyu-zh` | `modernbert-raid-mage`、`desklib-en`、`vanguard-en`、`aigc-env3` |
| `precise` 高精度（4 / 5） | `aigc-zhv3`、`aigc-zhv3short`、`simpleai-zh`、`xiaoyu-zh` | `modernbert-raid-mage`、`vanguard-en`、`aigc-env3`、`desklib-en`、`gltr` |

括号里是「中文档 / 英文档」的引擎数。为什么这么配：

- **`fast`** 只留"最快且互补"的两个：中文 `aigc-zhv3` + `simpleai-zh`（新一代 + 老牌各一票），英文 `modernbert-raid-mage` + `vanguard-en`（社交媒体向 + 正式文体向）。
- **`balanced`** 在快档上各补短板：中文加 `xiaoyu-zh` 做第三人交叉校验；英文加 `desklib-en`（文档级，长文更稳）与 `aigc-env3`（v3 新一代）。
- **`precise` 必须是 `balanced` 的超集**：中文四个分类器全上（`aigc-zhv3short` 专门覆盖短句分布）；英文 = 均衡四个 + `gltr`（方法型，对统计异常与学究腔敏感），共 5 个。**早期版本在英文档上把 `desklib-en` 换成了 `gltr`，导致均衡与高精度都显示 "4 引擎"、高精度看起来没生效**（2026-11-21 修正）。

解析规则（`ui/src/modelModes.ts` → `resolveModeEngines`）：

- **只从跑得动的里选**：未下载的引擎在选择期就被剔除（`usableEngineIds` = 非 router 且 `installed`），所以按钮上的「（N 引擎）」就是这次真正执行的引擎数，不会出现"显示了却跑不了"。
- **语言由文本决定**：前端按 CJK 字符占比做启发式（CJK 数 ≥ 拉丁字母数即判中文），**不调用**语言路由引擎——路由协议的响应里没有语言字段，而字符普查免费、确定、可离线。
- **`precise` 先确认语言**：进子页选「中文文本 / 英文文本」，因为高精度档要跑 4–5 个引擎，误判语言的等待代价最高；选择会记忆，下次沿用。子页每行直接标出「N 引擎 + 引擎名」，且按**可用集**计算，与按钮上的计数同源。
- **`cpu` 档剔除 GLTR**：方法型引擎要跑一次 GPT-2 基座前向，是 CPU 上最慢的尾巴；剔除后若集合为空，回退到该语言的快档集合（永不留空集）。
- **`fast-detectgpt` / `binoculars` 永不进自动档**：独占重型引擎（10.67GB / 28.87GB）必须由用户在自定义里显式勾选，不允许后台静默拉起来。
- **记忆**：模式 `aigc.detect.mode`、精确档语言 `aigc.detect.preciseLang`、自定义勾选 `aigc.detect.engines`，都在 localStorage。

## 自定义模式：手动勾选（二级面板）

- **入口**：主界面不再内联铺开勾选网格；点「自定义」后在胶囊旁弹出二级面板。主界面只剩胶囊与「开始检测」，两者都靠左，不会再被勾选网格挤到右侧。
- **面板形态**（2026-11-21 改版）：宽度固定 **372px**（比一级模式菜单窄），高度**按视口自适应**——贴住锚点一侧的可用空间，最高到 `视口高度 − 16px`，绝不越过窗口上/下缘；列表**内部滚动**，头部（返回 / 已选 / 批量）与底部「完成」不随滚动移动，所以「不要求一次看完」也不会丢掉操作入口。
- **每行结构**：左侧勾选框（自绘，勾选时方框填充、对勾以弹簧曲线弹入，按下有回缩反馈）；中右侧上行是**引擎名 + 文本类型标签**（中文 / 英文 / 多语），必要时追加「驻留」「未下载」；中右侧下行是**适用性小字**（最多两行，悬停可看全文）。语言从 `engine.lang` 来，说明文案来自 `modelModes.ts` 的 `ENGINE_NOTES`，不是营销词。
- 默认勾选 = **全部已安装且非 router 的引擎**（当前 12 个里装好的 9 个；随下载状态变化）；下载状态一到就自动剔除失效项，用户改动会持久化。
- **批量操作**：面板顶部一行「已选 N / M」+「全选 / 清空 / 反选」，一次写入完成——之前要"清掉 9 个再逐个勾回来"，最慢路径太长。批量只作用于可勾选集合（已安装、非 router）。
- 未安装的引擎**灰显**并标「未下载」；`r1-lang-router` 恒禁用（只做语言识别，不产出 AI 率）。
- 列表末尾另起一段**只读**的「自定义库（N）· 已登记，未接入推理」：列出 `models/custom/custom_models.json` 里的条目，标出文件数与体积、以及是否匹配到官方引擎。这些条目**不能勾选**——它们只是文件集登记，没有对应的推理实现；真正要参与检测，得按「新引擎接入五步」补上引擎类。放在这里是为了让"导入成功但选不到"这件事有明确答案，而不是让条目凭空消失。
- 注意引擎列表来自 `engines` 目录、自定义库来自 `custom/list`，是两套数据源；模型管理页把两者分两块呈现，选择模型面板则在同一张二级面板里先列引擎、后列只读库条目。

## 模型管理：默认集、风险与存储

| 引擎 id | UI 显示名 | 默认下载 | 风险 | 说明 |
|---|---|---|---|---|
| `simpleai-zh` | SimpleAI Chinese detector | ✅ | none | HC3 系语料训练 |
| `xiaoyu-zh` | Xiaoyu Chinese detector | ✅ | none | 与 simpleai 互补校验 |
| `aigc-zhv3` | AIGC Detector zh v3 (CN) | ✅ | none | v3 新一代，对齐 DeepSeek-V3 / GPT-4 |
| `aigc-zhv3short` | AIGC Detector zh v3 short (CN) | ✅ | none | 短句特化变体 |
| `modernbert-raid-mage` | ModernBERT raid-mage (EN) | ✅ | none | RAID 数据训练 |
| `desklib-en` | Desklib DeBERTa-v3-large (EN) | ✅ | none | 文档级，全篇一个分数 |
| `vanguard-en` | PAN vanguard ModernBERT (EN) | ✅ | none | PAN 2026 vanguard 血统 |
| `aigc-env3` | AIGC Detector en v3 (EN) | ✅ | none | v3 英文变体 |
| `gltr` | GLTR / perplexity (EN) | ✅ | none | 方法型，基座 GPT-2 |
| `r1-lang-router` | Language router (papluca xlm-r) | ✅ | none | 基建，不参与检测 |
| `fast-detectgpt` | Fast-DetectGPT (EN) | — | medium | 实现已接线，权重（GPT-Neo-2.7B）未下载，阈值未标定 |
| `binoculars` | Binoculars (EN, heavy) | — | high | 实现已落地，权重 28.87GB 未下载，4bit 未实现 |

- **「默认」只影响首次勾选**：默认集 = 除 `fast-detectgpt` / `binoculars` 外的 10 个；取消勾选不会影响已下载的文件，也不锁死任何引擎。
- **风险来自 manifest**（`risk_level`，D20）：`medium` / `high` 在勾选下载时弹**二次确认**弹窗并展示该引擎的具体代价（显存、延迟、失败条件）。`min_vram_mb` 也会随目录下发，但目前只用于文案提示，**没有硬拦截**——真正拦截在加载期（LRU 拒绝或 OOM 报错）。
- **体积口径**：本页用十进制 GB（manifest 字节数）；界面以 1024 为基显示，所以同一引擎会看到「390.7 MB」而非「0.41GB」，两者是同一个数，只是进制约定不同。
- **实测驻留**（RTX 5060 8GB 级，预算按卡换算 6.35GB）：7 引擎全驻峰值 3110MB；中文四引擎（precise）共驻 1060MB；三引擎热态出结果 0.059s。单个分类器实测 **202MB**（registry 声明 220 / 400MB，属保守估算）——界面上的「实测/估算」标记来自 `torch.cuda.memory_allocated()` 前后差值，见[架构设计 §4](ARCHITECTURE.md)。驻留时长由设置里的留存档位控制（常驻 / 30·10·5 分钟 / 不驻留）。
- **存储**：每个引擎可选 `keep-original`（默认）/ `dual` / `fp16-only` 三档，模型管理里能看到「需转换」状态——fp32 在盘但 fp16 载荷缺失时走**转换**，绝不重下载（见[架构设计 §5](ARCHITECTURE.md)）。
- **自定义模型库**：支持本地导入已下载的 HF 目录（文件集命中官方引擎时硬链接/复制并立即变「已就位」）与 URL 直连下载；未命中官方引擎的目录登记进 `models/custom/<name>/`，留给后续版本接线。库表列 `名称 / 来源 / 文件数 / 体积 / 说明 / 时间`；**时间按本机时区显示**（后端存 UTC ISO-8601，前端换算——直接截字符串会把 00:29 显示成前一天 16:29）。命中官方引擎的条目会在名称后标「（官方引擎 id）」。

## 验收口径

每个分类器上线前跑两组探针：

1. **严格单样本三探针**：懒加载（LRU）、标签方向（AI 高分 / 人类低分）、分句 offset 对齐 + 卸载显存释放。
2. **多样本电池**（`scripts/probe_battery.py`）：4 条人类 + 3 条模型文本，要求 `max(AI) ≥ 0.9`、人类中位数 `< 0.5`、人类最大值 `< 0.7`。原始数字落在 `VERIFICATION/P1-<engine>-battery.json`。

一条文本的单次判定只能算信号，不应当作结论——这是方法型引擎与短文本分类器的共同边界。

## 已知限制（如实标注）

- **HC3 系正式文体过判**：`simpleai-zh` / `aigc-zhv3` 的训练分布来自 HC3 系语料，维基/论文腔的人类文本可能被抬高。`zhv3short` 对极短人类句偶发 0.60；`env3` 对口语/邮件体偶发 0.59（电池实测）。
- **desklib-en 是文档级引擎**：整篇一个分数，句级切片属于分布外（OOD），会制造误报。它与句子级引擎配对使用。
- **GLTR 截断**：超过 1024 个 GPT-2 token 的文档尾部不参与打分。
- **Fast-DetectGPT 阈值未标定**：实现已接线（采样只统计生成 token、原文/样本跨度对齐），但 `CENTER/SCALE` 是临时值，重验通过前分数不可信；基座 GPT-Neo-2.7B 需 10.67GB 下载。
- **Binoculars**：fp16 双 7B 约 28GB 显存，独占运行；4bit 量化未实现（运行时没有 bitsandbytes，其 cp314 轮子未验证）。8GB 显卡无法运行，UI 会明确警告。
- **EnsemJudge（未入目录）**：官方系统是 18 投票成员的集成框架（含 GLM-4-9B / Qwen2.5-7B LoRA），需要 2×V100，且代码仅授权比赛评测用途、权重仓库未声明许可。仓库内只有**框架 + 规则投票器**的实现（`backend/engines/ensemjudge.py`），不冒充完整系统，也暂不提供下载入口。

## 新增引擎的接入流程

1. `scripts/gen_manifest.py` 加条目 → 生成 manifest（SHA 锚定 HF LFS oid）
2. `backend/engines/registry.py` 加 `EngineSpec`
3. 有自定义结构 → `custom_models.py` 写本地复刻；标准分类头直接用 `RealClassifierEngine`
4. `scripts/fetch_engine.py <id>` 下载 → `scripts/probe_classifier.py` / `probe_battery.py` 验收
5. 通过后 `verified=True`，并在前端 `modelModes.ts` 补充说明与模式配置
