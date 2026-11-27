# 架构设计

> 本文面向想了解实现细节的开发者。产品定位与功能列表见 [README](../README.md)。

## 1. 进程模型

```
┌─ Tauri 壳（Rust，~100 行，只做保姆）──────────────┐
│  spawn sidecar · 读 stdout 拿端口 · env 发 token  │
│  asset serve 前端 · 关窗 kill 兜底                │
└───────────────┬──────────────────────────────────┘
                │ 控制（一次性）：spawn / 端口 / token / kill
                ▼ 不在数据路径上
前端 WebView ←──── WS ws://127.0.0.1:{port} ────→ Python sidecar
(HTML/CSS/JS)      msg_id 关联 · 流式分帧          │
                                                torch → GPU / CPU
资源目录 %LOCALAPPDATA%\AI-Text-Detector\
  ├─ runtime\python\   (embeddable Python)
  ├─ site-packages\    (torch / transformers)
  └─ models\           (HF 目录结构 + manifest.json)
```

为什么 Rust 不在数据路径上：WebView 里的 JS 拿不到子进程的 stdio，若用 stdio 当数据通道，所有消息都要 Rust 永久中转，等于双倍 IPC 税。WS 直连让壳保持"只负责生和杀"。

## 2. 通信协议

统一信封：

```json
{"v": 1, "id": "m-001", "type": "req|res|event|ping", "ts": 0, "payload": {}}
```

- 请求/响应靠 `id` 关联；进度与流式结果走 `event`（`progress` / `partial` / `done`）
- 健康检查：5s 间隔 ping/pong，断线后前端自动重连（1s 起步）
- 大块二进制禁止 base64，只传路径或分帧（例外：浏览器无路径可用时，导入文档以 base64 走 `file/import`，单文件 20MB 上限）
- **原文不出后端**：检测结果只回传 `[offset, length, score]` 数组与整篇率，不回传文本副本。
  **唯一例外（D40）**：`detect` 带附件时，回复的 `input.text` 会回传拼好的全文——文件正文只存在于 sidecar，而结果区的逐句列表与高亮必须拿到真实句子；纯文本检测仍守原契约

主要消息族：`health`、`runtime/status`、`catalog`、`detect` / `detect_part`、`download/*`、`custom/*`（本地导入 / URL 下载）、`storage/*`（含 `storage/uninstall`）、`resident/*`、`file/*`（文档导入）。

**文件导入（D40）**：`file/import` 接受 `{paths:[...]}`（Tauri，原生选择器给绝对路径）或 `{blobs:[{name,data(base64)}]}`（浏览器 fallback），
逐文件解析后把正文留在会话表里，只回 `{id,name,ext,chars,bytes,notes}`；`file/drop` 删一个，`file/list` 重读（前端在连接建立后同步一次，避免重连后 UI 与 sidecar 不一致）。
解析器见 `backend/filereader.py`：docx 走 zipfile + ElementTree 读 `word/document.xml`（w:p 段落、w:tab、w:br），不依赖 python-docx/lxml；
pdf 用 pypdf 逐页 `extract_text()`，单页异常不拖垮整篇，提取不足 20 字判定为扫描件；txt/md 做编码探测（BOM → utf-8 → gb18030 → utf-8 replace）。
`detect` 的 `attachments` 顺序即拼接顺序，**文字在前、文件在后**，段间空行分隔。

**检测生命周期（round-19）**：`detect` 是**单飞**的——一次只跑一轮，第二个请求直接回 `a detect run is already in progress`
（排在 GPU 锁后面只会表现成卡死）。四条实现约束：

- `detect` 跑在**独立 task** 里，消息循环始终可读。旧实现把 `to_thread` 直接 await 在 `async for` 体内，
  循环被占满整轮，`detect/cancel`、`ping`、`health` 只能等这轮跑完才被读到——注释里“保持响应”的承诺从未成立。
- `detect/status` 回 `{job:{state, job_id, engines, current, chars, started_at, finished_at, elapsed_s, cancel_requested, reason, detail, has_result}}`，
  **不含正文**。重载后的页面靠它知道「我不在的时候上一轮做了什么」。
- `detect/cancel` 置协作式取消标志：worker 在**句子批次之间与引擎之间**检查（一次 torch 前向不可打断），
  已取消的运行一律回 `{"cancelled": true}`，不谎报 done（`finish` 里也会把「取消后仍跑完」的 done 改判成 cancelled）。
- 连接断开（重载 / 关窗 / 崩溃）**不再立即取消**（round-20 宽限接管）：只有断开者是当前活跃连接时才排
  `_reap_orphan(job_id)`，等 `DETECT_GRACE`（默认 3s）后再判定——期间有新页面接管（`ACTIVE_WS` 非空）就放行、
  `job_id` 已被新一轮替换就不动，否则 `cancel("disconnect")`。接管者自己没发起 detect，所以判定不依赖 `detect_task`。
- 跑完的回复若发不回原 socket，**只有真结果**（payload 带 `results`）进 `DetectJob.stash_result`；
  新页面用 `detect/result` 一次性取走（读走即清，`RESULT_TTL` 600s 过期），纯文本运行连原文一起暂存以便高亮视图可画。
  取消/错误回复不 park——`status` 的 `state`+`detail` 已经说明结局。
- 接管判定看对端 socket 状态：上一个连接已关闭时新连接直接接管，不再白等一轮重连退避。

**界面存活（round-19）**：壳内屏蔽 F5 / Ctrl+R / Ctrl+Shift+R（一次重载就断掉唯一持有 detect 回复的 socket，而白屏事故正好发生在两次重载之后）；
`work` 视图在 catalogue 未到位时显示**启动门**（计数，12s 后转「重试连接 / 重新加载界面」面板），不再留一片空白；
托盘图标提供「显示窗口 / 重新加载界面 / 退出」——`decorations:false` 下原生菜单栏会压在自绘标题栏上，所以走托盘而不是菜单条。

**驻留控制**（round-10）：`resident/policy` 读改写留存档位、`resident/unload` 立即释放（可指定 id，省略即全卸）；服务端主动推 `resident` 事件（空闲清扫器卸完模型后），前端收到即刷新。`engines` / `detect` / `resident` 三类报文统一携带同一个 `vram` 块：

```json
{"resident": 2, "resident_detail": [{"id": "aigc-zhv3", "name": "AIGC Detector zh v3", "mb": 202, "measured": true, "idle_min": 3}],
 "used": 404, "budget": 6499, "total": 8120, "measured": true, "retention": "default", "ttl": null}
```

口径：`used` = 各驻留引擎的 `actual_vram_mb` 之和（无实测时回退 `est_vram_mb`），`measured=false` 表示这批数字全是估算；`budget` 由显卡总量换算（见 §4）。

## 3. 检测流水线

**模式解析发生在前端，不在后端**：自动档按「语言 + 硬件档」从预设表里取引擎 id，再按「已安装 + 非 router」过滤；自定义档直接用用户的勾选。sidecar 收到的 `detect` 请求里的 `engines[]` 就是最终执行集，后端不再增删——所以界面上的「（N 引擎）」和实际执行永远一致。预设表与取舍理由见[引擎目录](ENGINES.md)。

1. **语言判定**：中文按 CJK 字符占比启发式路由；英文走 en 引擎组；语言路由引擎（XLM-R）作为多语言扩展地基保留
2. **分句**：`SENT_RE` 统一切分，引擎层与前端共用同一套边界；offset 是 Python code point，前端按 code point 切片（emoji 不会错位）
3. **逐句打分**：分类器按句批量推理（fp16 / fp32 由存储状态决定）；方法型引擎做整篇一次前向 + 句级 span 统计
4. **聚合**：
   - 句级高亮 = 多引擎 **max**（宁可误报不漏报），悬停显示各引擎分数
   - 整篇 AI 率 = 句长加权平均 `Σ(score×len)/Σ(len)`，文档级引擎的单段结果自然退化为原分数，零特判
5. **流式**：引擎每完成一批句子上报一次 `partial`，前端渐进高亮

### 逐句分是怎么来的（判断逻辑）

用户问「这一句凭什么算 82%」，完整链条只有四步，每一步都能在代码里对上：

1. **切句**：`SENT_RE = [^\s.!?。！？\n][^.!?。！？\n]*(?:[.!?。！？]+|$)`，中英标点同一套，引擎侧与前端共用，所以句子边界不会错位。空白句丢弃。
2. **打分**：
   - **分类器类**（aigc-zhv3 / zhv3short / env3 / simpleai / raid-mage / vanguard / xiaoyu）：句子本身就是模型输入，输出一个 0–1 的 AI 概率。
   - **方法型**（GLTR）：整篇只做一次前向（continuation 模式，不逐句孤立打分——句首没有上文会让困惑度坍缩），再按每句的 token 区间统计 top-k 排名占比；token 数 < 8 的句子给不出可靠统计，直接跳过（缺席 ≠ 低分）。
   - **文档级**（desklib / binoculars）：整篇一个分数，句级切片属于分布外，所以单引擎视图里它的整篇段被过滤掉，不参与逐句上色。
3. **聚合**：综合模式下每句取所有成功引擎的 **max**（保守聚合，宁误报不漏报）；单引擎模式下就是该引擎自己的分。每句背后的各引擎分数是**点开**看的（句子行的百分比就是按钮），不再依赖悬停。
4. **上色**：`bucketColor(score, mode)` 按当前高亮模式分桶——六度渐变阈值 0.90 / 0.75 / 0.60 / 0.45 / 0.25；简单三度按 0.75 / 0.50；极简两度按 0.50。分数是连续的，分桶只影响颜色，不影响百分比数字。

两个容易误解的点：**百分比是 AI 概率，不是置信度**；**「3/3 句」是可见行数 / 可打分句数**，引擎跳过或失败的句子不会凑数也不会染色。

### 界面动效（2026-11-21）

模式弹层的入场上浮、子页展开与关闭都走**弹簧曲线**，参数沿用 SwiftUI 的语义：`response` 决定手感快慢，`dampingFraction` 决定回弹（1.0 = 临界阻尼无过冲）。实现上由 `ui/src/spring.ts` 在启动时按解析解采样成 CSS `linear()`，写入 `--spring-pop` / `--spring-row` / `--spring-close` 三个变量——CSS 关键帧与 JS 退场共用同一份物理定义。

- 入场：`response 0.32s / damping 0.72`，512ms 采样窗，轻微过冲
- 列表行与子页：`0.30s / 0.78`，带一点惯性
- 退场：`0.26s / 0.92`，几乎不回弹，点空白后**先播完退场再卸载**（约 420ms），不再硬切
- 卡片 / 列表展开：`0.42s / 0.76`（`--spring-expand`，672ms）——质量更大、过冲只一点点，收放都不急停
- 百分比按钮悬停：`0.24s / 0.86`（`--spring-menu`）抬起 1.05 倍 + 微阴影，按下回缩 0.94
- 侧栏切页（工作台 / 模型管理 / 设置）：旧面板 140ms 上浮 8px + 淡出，新面板 `0.30s / 0.78` 从下方 10px 弹回、260ms 淡入

切页是**两段式**：`view` 管导航高亮与数据副作用（点击当帧生效），`shown` 管真正绘制（落后一个退场动画）。只做纵向位移——920px 内容列在 ~1180px 窗口下横向挪 16px 就会越界闪出横向滚动条。

**fill 只能 `backwards`，不能 `both`**：`both` 会把"最后一帧"永久留在元素上，计算值停在 `matrix(1,0,0,1,0,0)` 而不是 `none`；任何非 `none` 的 transform 都会成为 `position: fixed` 子元素的包含块，于是模型胶囊、逐句百分比弹窗会以面板为原点定位、整体错位。

展开/收起不是定时长过渡再硬切：`Collapse` 用 `grid-template-rows: 0fr → 1fr` 让浏览器自己插值高度（无需 JS 量高），弹簧曲线带轻微过冲，所以面板有"质量感"；关闭时内容**保持挂载**，动画播完才归零，中途反向不会跳。

时长必须与采样窗口一致，否则曲线被时间缩放，回弹会变成慢漂移。

勾选面板（自定义）单独一套尺寸规则：宽 **372px**，高度取锚点上/下两侧**更宽裕的一侧**，上限 `视口高度 − 16px`，因此矮窗口不会顶出屏幕、高分辨率下自然拉长；列表内部滚动，头部与「完成」按钮固定。勾选框是自绘方框，选中时填充并把对勾从 0.45 缩放到 1（`--spring-row`），按下时整框回缩到 0.86，松开弹回——原生 checkbox 无法承载这段曲线。

**布局**：固定侧栏占 242px（12 + 218 + 12），内容列上限 920px，左边距取 `max(242px, 50% − 339px)`——宽屏时在「扣掉侧栏后的可用区」里居中（1600 窗口：左右各留 219px），窄屏时自动贴住侧栏、宽度跟着缩，不会钻到侧栏底下。刻意不用 `transform` 居中：它会给 `position: fixed` 的子元素造出新的包含块，右键菜单和胶囊弹层会全部错位。

### 结果区信息架构（2026-11-21）

自上而下一条线，越靠上越"结论"，越靠下越"证据"：

1. **综合卡**（`CompositeCard`）：大号整篇 AI 率 + 分桶标签（几乎确定 AI / 疑似 / 不确定 / 大概率人工 / 几乎确定人工）+ 一行覆盖说明（N 个引擎 · M 句 · 逐句取最高分后按句长加权）。左下角三角形**弹簧展开**详情表（见下方「吸顶与展开」；三角形是 `viewBox` 矢量 path 而非 `▸` 字符——10px 文本字形会因字体回退丢掉上半边）。
2. **控制行**：两个胶囊——「筛选」（全部句子 / 只看 AI 率 / 只看人工率）与「模型」（综合，即全部引擎 max；或单个引擎）。胶囊是自绘组件而非原生 `<select>`，选项带说明、选中项打勾，弹出走 `--spring-pop`。
3. **高亮正文**：按当前筛选与模型上色，图例在下方。**不再有悬停提示**——正文保持可选可复制，没有会突然冒出来的浮层。
4. **失败引擎框**：仅在引擎报错时出现，逐个给原因；成功的引擎不再单独罗列——各引擎的整篇率在「模型」胶囊的选项说明里（`整篇 77%`），完整分数表在综合卡的「计算详情」里，信息不重复铺两遍。
5. **逐句明细**（`SentenceList`）：与高亮同源的数据，**默认收起**（只有一行表头 + 「展开」按钮），行 = 序号 / 引文 / 百分比按钮。百分比是按钮不是标签：悬停按弹簧曲线抬起一点，**点击弹出二级窗口**列出「哪个模型给了多少」，行内按概率降序、带各自色块，底部一行口径说明；Esc / 点空白 / 滚轮关闭，退场播完才卸载。

**吸顶与展开**：综合卡在结果面板里 `position: sticky; top: 0`，无论面板多高都不会被滚走；卡片顶部有一块 `::before` 盖住滚动容器 10px 的 padding，滚动时文字不会从卡片上沿透出来。**「放大」不再是"把滚动交给整页"**，而是把结果面板的滚动区从 560px 拔高到 72vh——因为面板本身 `overflow: hidden`，页面级滚动会连卡片一起带走，sticky 只能对内部滚动器生效。两个状态共用同一个滚动器和同一条吸附规则。

**排序规则集中在 `sentenceRows` 一处**：`只看 AI 率` → 分数降序（最像 AI 的在最上面）；`只看人工率` → 分数升序（最像人工的在前）；`全部句子` → 跟随设置里的「原文顺序 / 按 AI 率」开关。**模型胶囊的选择同时作用于高亮与逐句**，不出现"上面看 A 模型、下面看 B 模型"的错位。

输入区的放大按钮把文本框切到 `.composer.tall`（`min-height: min(58vh, 640px)`，430ms 过渡），只改高度不改布局结构，长文粘贴时阅读面积翻倍；引擎条与放大按钮同排置顶，避免按钮散落在正文区。

**输入条上的引擎选择**（round-12）：原先把每个引擎名平铺在一行，选到 5 个引擎就被截断成「…DeBERTa-…」。现在只留一枚「本次 N 引擎 ▾」胶囊，点开是二级窗口：引擎名 + 语言标签（中文/英文/多语）+ 来源说明（来自哪个预设、要改请点模式胶囊）。它和「筛选 / 模型」共用同一套弹层机制（`Popover`：fixed 定位、视口翻转与钳制、弹簧入场、WAAPI 弹簧退场），只是不带选中语义。

### 右键菜单（2026-11-21）

自绘，接管 `contextmenu` 事件后 `preventDefault`：四角 12px 圆角、半透明白底（0.985，避免入场动画中间帧发灰）、`--spring-menu` 弹簧入场、退场走 WAAPI 的 `SPRING_CLOSE` 且**播完再卸载**。fixed 定位 + 视口翻转钳制（下方不够就向上翻）。键盘：↑↓ 移动、Enter 执行、Esc 关闭；外部 mousedown、滚轮、窗口 resize 均关闭。条目固定：剪切 / 复制 / 粘贴 / —— / 全选，可用性按当前是否有焦点输入框与选区实时判定（无选区时剪切、复制置灰）。

每次打开用 `seq` 作 React key 强制新实例——否则同一坐标连开两次会复用正在退场的旧节点，出现"点了没反应"。

**粘贴不需要授权**：Tauri 壳内**原生插件优先**（`plugin:clipboard-manager|read_text`），`navigator.clipboard` 只作浏览器验证时的兜底——WebView2 对 `navigator.clipboard.readText()` 会弹「此页面想读取剪贴板」，那正是桌面应用不该有的网页味。写回仍走 `execCommand`，保留 undo 栈，Ctrl+Z 能撤回粘贴。

**非文本内容必须说出来**：Ctrl+V 由 webview 自己处理，所以用 DOM `paste` 事件的 `clipboardData.types` 判断内容形态（这个不需要任何授权）：没有 `text/plain`、却带 `Files` 或 `image/*` 时 `preventDefault` 并记下原因（文件 / 图片 / 其他）。输入框仍为空时点「开始检测」，弹「内容格式不支持」模态说明为什么什么都没粘进来；确认后清除标记，按钮回到禁用。右键菜单的「粘贴」由我们自己读剪贴板，读到空文本同样落成 `nontext` 标记。

### 侧栏设备卡片（2026-11-21 round-14）

侧栏底部一张 188px 宽的小卡，四行：CPU（型号 + 逻辑线程）、内存、显卡、显存——显存写的是**进程实际能请求到的总量**（8GB 卡报 7.93 GB，与预算 6.35 GB 同源）。数字全部取自 sidecar 启动时的 `hardware.collect()`；前端不二次探测，否则面板与引擎预算会来自两个来源，出现「显示 8 GB、预算 6.35 GB」这种看着自相矛盾的画面。CPU 型号必须读注册表 `ProcessorNameString`：`platform.processor()` 在 Windows 只返回 `AMD64`，而注册表值带尾部填充空格和双空格，要 `split/join` 归一化。

## 4. 显存与加载策略

**预算按卡换算，不写死数字**：`default` = 显卡总量 × 0.8（RTX 5060 8GB 级 → 6499MB），`forever` = 总量 − 512MB 系统预留（→ 7612MB），探测不到卡才回落 6144MB。留两成空头寸是因为 WebView2 合成、CUDA context 与 torch 缓存碎片都会占一点，卡满即 OOM。

- **LRU 淘汰**：超预算时先卸「语言不在本次运行集合内」的引擎（本次跑 5 个英文引擎 → 中文引擎先出局），同语言内再按最久未用。这样跨语种切换不会互相挤掉热模型。
- **独占引擎**（Binoculars 类）先卸空其他引擎，允许超过预算独自运行。
- 非独占引擎若**估算**显存超过整个预算，直接拒绝（配置错误，不做无意义的 OOM 尝试）。
- 硬件分三档（CPU / 标准 / 充裕），只影响默认勾选与警告文案，**永不锁死引擎选项**。

**显存数字是实测的**：`load()` 前后各取一次 `torch.cuda.memory_allocated()`，差值写进 `actual_vram_mb`。只认「加载后比加载前多」的情况——no-op 加载（模型已在驻留）差值为 0，不能把已测到的真实值抹掉。无 GPU 或未量测时回退 registry 的 `est_vram_mb`，界面对应标「估算」。

**留存六档**（设置 → 常规）：

| 档位 | 行为 |
|---|---|
| `常驻(永久)` | 永不因空闲卸载，只能手动释放 |
| `常驻(默认)` | 同永久，但重启后回到默认档 |
| `30min` / `10min` / `5min` | 空闲超过 1800 / 600 / 300 秒由清扫器卸载 |
| `不驻留` | 检测批次结束立刻全卸 |

清扫器 30s 一跳，与检测共用同一把 `asyncio.Lock`，避免检测跑到一半模型被抽走。`不驻留` **不能交给清扫器**——同批引擎会互相把对方判成"空闲"然后连环卸载，必须由批次收尾统一释放。

本机实测（RTX 5060 8GB 级）：7 引擎全驻峰值 **3110MB**；中文四引擎（高精度档）共驻 **1060MB**；三引擎热态出结果 **0.059s**；两个中文引擎实测各 **202MB**（声明值 220 / 400MB，偏保守）。

**下载源两档**（设置 → 常规，D36）：`优先使用官方源`（默认，huggingface.co 首站）与 `优先使用镜像源`（hf-mirror.com 首站）。只影响「模型管理 → 引擎下载」，另一源永远是失败兜底；自定义模型库的 URL 下载保留自己的镜像询问。

## 5. 存储策略

每个引擎可选三档（`models/storage.json` 持久化）：

| 模式 | 磁盘 | 行为 |
|---|---|---|
| `keep-original`（默认） | fp32 原件 | 加载时现场转 fp16，SHA 校验链完整 |
| `dual` | fp32 + fp16 副本 | 加载免转换，副本名 `model.fp16.safetensors` |
| `fp16-only` | 仅 fp16 | 转换校验后覆盖原件，省约一半磁盘；fp32 加载不可用 |

`cast-index.json` 记录 fp16 载荷的 SHA256 与来源 fp32 的 SHA256，是 fp16-only 状态下的完整性锚点——否则下载器会把"原件缺失"误判成需要重新下载。

磁盘状态派生出四个标志，模型管理里的「已就位 / 需转换 / 可清理」就是它们：

| 标志 | 含义 |
|---|---|
| `installed` | 按**当前模式**是否完整：`keep-original` 看 fp32 原件，`dual` 要两份都在，`fp16-only` 只看 fp16 载荷 |
| `recast_needed` | fp32 在盘但 fp16 载荷缺失/损坏 → 跑转换任务（不是重下载） |
| `stale_cast` | 档位是 `keep-original` 却存在 fp16 载荷 → 提示可清理释放空间 |
| `dtype_options` | 这次磁盘状态允许的加载精度；`fp16-only` 下 fp32 加载不可用 |

模型目录布局：

```
models/
  manifest.json        下载清单：url/sha256/字节数/风险等级/最小显存
  storage.json         每引擎存储档位（默认档不写入）
  <engine-id>/         HF 目录结构 + cast-index.json（fp16 载荷索引）
  custom/              自定义模型库：custom_models.json + <name>/
```

`download/status` 一次返回「下载进度 + 磁盘派生状态 + 硬件档位」，所以界面不需要自己拼装状态——前端只做展示与文案。

## 6. 下载器（五铁律）

1. 多源回退（链路首站由「设置 → 常规 → 下载源」决定：官方优先 / 镜像优先，另一源恒为兜底）
2. Range 断点续传（`.incomplete` 中间态）
3. SHA256 校验（锚点来自 HF LFS oid），不匹配拒绝落盘
4. 磁盘空间预检
5. 原子改名（`os.replace`），不留半成品

源偏好（D36）只在任务启动时重排一次模板链（`downloader.templates_for`），整条任务共用同一顺序：`download/start {engines, source}` → `download/status.source` 回显 → `download_engine_start` 事件带 `source`。偏好存前端 localStorage，后端保持无状态；自定义模型库的 URL 下载不走这条链，它有自己的「是否走镜像」询问。

进度只报真实字节数：样本不足显示 `--`，估算值标 `≈`。

## 7. 安全模型

威胁模型是**本机回环**：sidecar 只监听 `127.0.0.1`，连接需一次性 token（常量时间比较）+ Origin 校验；第二连接直接拒绝。不引入 MAC / seq / replay 全套——那是敌意宿主页的配置，与本项目的本地工具定位不匹配。

## 8. 目录结构

```
vX.Y.Z/            版本工作区（代码 / 测试 / 验收）
  backend/         server.py · engines/ · downloader.py · protocol.py
  ui/              前端（TS + Vite）
    App.tsx        状态与装配（模式、结果、设置）
    CompositeCard / SelectCapsule / SentenceList / ContextMenu   结果区与通用件
    spring.ts      弹簧曲线采样 → CSS 变量（--spring-*）
    clipboard.ts   剪贴板桥（navigator.clipboard → Tauri 插件回退）
  src-tauri/       Tauri 壳
  tests/           pytest（fixtures + 契约测试）
  models/          权重（git 忽略）+ manifest.json
  VERIFICATION/    本版本验收工件
docs/              跨版本文档
```
