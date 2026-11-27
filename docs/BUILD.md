# 构建与开发

> 面向开发者。架构与协议见[架构设计](ARCHITECTURE.md)，引擎清单见[引擎目录](ENGINES.md)。

## 环境要求

| 组件 | 版本 | 说明 |
|---|---|---|
| Windows | 10 / 11 | 目标平台 |
| WebView2 | Evergreen | Tauri 前端运行时，系统一般已带 |
| Python | **3.14** | sidecar 与全部脚本（cp314 轮子已验证） |
| Node.js | 22 | 前端构建 |
| Rust | stable | Tauri 壳 |
| NVIDIA 驱动 | 随卡 | 无卡机器自动回落 CPU |

## 一次性准备

```bat
cd v0.2.0-P1
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install torch==2.9.0+cu129 --index-url https://download.pytorch.org/whl/cu129
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-dev.txt

cd ui && npm install
```

Tauri CLI（仅首次）：`cargo install tauri-cli --version 2.12.4`

## 开发启动

两个终端：

```bat
:: 终端 1 - 前端 dev server（1420）
cd ui && npm run dev

:: 终端 2 - Tauri 壳（会 spawn sidecar 并注入端口/token）
cd src-tauri && cargo tauri dev
```

只想调后端时，可以单独起 sidecar（开发令牌 + 额外 Origin）：

```bat
set SIDE_TOKEN=dev-token
set SIDE_EXTRA_ORIGINS=http://localhost:1420
.venv\Scripts\python backend/server.py
```

端口写在 stdout 的引导报文里；前端「设置 → 开发者选项 → 连接设置」可手填端口与 token。

Windows 调试构建默认隐藏 Tauri 控制台；可在「设置 → 开发者选项 → 启动与调试」
中打开或关闭独立的只读日志窗口。关闭日志窗口不会关闭主应用。设置持久化到
`%LOCALAPPDATA%\com.aitextdetector.app\console-visibility.json`。
需要强制显示时设置环境变量 `AIGC_SHOW_CONSOLE=1`。Python sidecar 始终使用
`CREATE_NO_WINDOW` 启动，日志仍写入 `logs/`。

## 测试与质量门禁

```bat
.venv\Scripts\python -m pytest tests/ -q          :: 单元 + 契约测试
.venv\Scripts\python -m ruff check backend/ tests/
.venv\Scripts\python -m mypy backend/
cd ui && npm run build                            :: tsc + vite

:: 提交前泄露自查：本机绝对路径 / 用户名 / 主机名 / IP / 密钥 / 邮箱
:: 扫描范围 = git 会发布的文件（tracked + 未忽略的 untracked），有命中即 exit 1
.venv\Scripts\python scripts/leak_scan.py
```

## 模型管理

模型清单 `models/manifest.json` 是下载器的唯一真源（每个文件带 HF LFS 的 SHA256 与字节数）。

```bat
:: 重新生成清单（会与远程 LFS oid / 本地文件对拍，不一致直接中止）
.venv\Scripts\python scripts/gen_manifest.py

:: 按引擎下载缺失文件（多源回退 / 断点续传 / 校验 / 原子落盘）
.venv\Scripts\python scripts/fetch_engine.py aigc-zhv3

:: 验收：严格单样本三探针 + 多样本电池
.venv\Scripts\python scripts/probe_classifier.py aigc-zhv3
.venv\Scripts\python scripts/probe_battery.py aigc-zhv3
```

验收工件落在 `VERIFICATION/`。

## 存储模式

`models/storage.json` 记录每个引擎的存储档位（`keep-original` / `dual` / `fp16-only`）。切换在「模型管理 → 存储管理」里操作；fp16-only 会覆盖 fp32 原件，之后切回需要重新下载（UI 有明确提示）。

## 打包

当前 release 资源树包含 `backend/` 与 `models/manifest.json`，Tauri 配置使用 NSIS
current-user 安装，并启用 VC++ runtime 与 WebView2 bootstrapper。

发布构建必须使用路径重映射包装脚本，避免 Rust panic location 把构建机的用户目录
和 Cargo registry 绝对路径写进二进制：

```bat
python scripts/build_release.py --bundles nsis
```

包装脚本会为 Cargo registry、项目目录和用户目录注入
`--remap-path-prefix`，其余参数原样转发给 `cargo tauri build`。

运行时分成三层，不能混成一个不可拆的大包：

1. **主体包**：Tauri shell、前端静态资源、embeddable Python、`websockets`、后端源码和 manifest。
2. **推理运行时**：PyTorch、Transformers、safetensors、tokenizers 等，按目标机选择 cu129 或 CPU。
3. **模型权重**：用户在模型管理中按需下载。

CPU 版与 CUDA 版 PyTorch 的检测口径、下载入口和切换注意事项见
[推理运行时与 CPU 版 PyTorch](RUNTIME-COMPONENTS.md)。

主体包的可启动性已有可重复验收脚本：

```bat
python scripts/audit_base_runtime.py
```

该脚本会临时搭建一个不含 torch / transformers / 模型的 embeddable runtime，
启动真实 sidecar，并完成授权 WebSocket ping 与 `runtime/status` 检查。
2026-11-27 实测主体 runtime 为 26,176,985 B（不含 Tauri exe 和压缩），
`local_service_ready=true`、`inference_ready=false`，证明“界面与后端 WS 可先启动，
推理运行时后装”的拆分可行。

主体包体积实测（含 release exe、前端、Python、websockets、backend、manifest 与 VC++ DLL，
不含 torch / transformers / safetensors / 模型权重）：

- NSIS 实测 **12.64 MiB / 13,252,530 bytes**
- 包内不包含 `__pycache__`、`.pyc`、PyTorch 或模型权重

WebView2 使用 `downloadBootstrapper`，完整 WebView2 Runtime 不计入主体安装包。

发布说明应同时保留[版本状态与已知限制](RELEASES/v0.2.0-P1.md)，其中明确了
`pre` 表示概念验证阶段，以及当前测试环境、硬件覆盖和功能完成度限制。
