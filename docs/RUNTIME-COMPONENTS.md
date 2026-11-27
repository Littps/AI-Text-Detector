# 推理运行时与 CPU 版 PyTorch

> 本文说明「基础服务组件 → 推理引擎」中 CUDA 版与 CPU 版 PyTorch 的检测口径。
> 主体包、模型权重和推理运行时的边界见[构建与开发](BUILD.md)。

## 目标

同一台机器只需要一组可工作的 PyTorch 轮子：

- NVIDIA 驱动和对应 CUDA 构建可用时，优先使用 CUDA 版；
- 没有可用 CUDA、驱动版本不匹配或需要节省显存时，可以选择 CPU 版；
- 两个轮子不能同时作为“必须安装”的组件，否则会把 CPU 回退误报成缺依赖。

因此页面会额外显示 **PyTorch CPU 版** 卡片。它通常是可选替代
（`required=false`）；只有硬件推荐 CPU 且当前 PyTorch 无法导入时，
它才会标成影响检测的必选回退。
它不会改变当前 sidecar 已经加载的 Python 进程，也不会在检测运行中替换
原生 DLL。安装或切换轮子后，需要重新启动应用让 sidecar 重新导入。

## 判定矩阵

| 当前状态 | `torch` 卡片 | `torch-cpu` 卡片 | 推理就绪 |
|---|---|---|---|
| CPU 版导入成功 | 正常 | 正常，显示 CPU 版 | 其他推理依赖齐全时是 |
| CUDA 版导入成功、CUDA 可用 | 正常 | 需注意，作为可选替代 | 其他推理依赖齐全时是 |
| CUDA 版存在、驱动不可用 | 需注意 | 需注意，建议 CPU 回退 | 其他推理依赖齐全时是 |
| PyTorch 包存在但导入失败 | 缺失 | 缺失或需注意 | 否 |
| PyTorch 未安装且硬件推荐 CPU | 缺失 | 缺失，指引 CPU wheel | 否 |
| PyTorch 未安装且硬件推荐 CUDA | 缺失 | 需注意，CPU 仍可作为回退 | 否 |

`runtime/status` 同时返回 `torch_variant`：

```json
{
  "torch_variant": "cpu",
  "components": [
    {"id": "torch", "state": "ready"},
    {"id": "torch-cpu", "state": "ready"}
  ]
}
```

`torch_variant` 的取值是 `cpu`、`cuda` 或 `unavailable`。它是展示字段，
不是新的安装状态；真正的就绪判断仍由 `inference_ready`、导入探测和其他
推理包版本检查共同决定。

## 下载入口

「基础服务」的缺失项列表按照用户设置的下载源偏好选择入口：

- 官方 CPU wheel 索引：<https://download.pytorch.org/whl/cpu>
- 镜像兜底：<https://pypi.tuna.tsinghua.edu.cn/simple/torch/>
- CUDA wheel 索引：<https://download.pytorch.org/whl/cu129>

页面只负责打开受 allowlist 限制的下载页；不会从任意 URL 直接执行安装脚本。
开发环境可以按当前 Python 版本选择轮子后手动安装，例如在已准备好的虚拟环境
中使用 pip；不要用系统 Python 覆盖应用托管的 runtime。

安装后回到「基础服务组件」点击「重新检测」。如果是应用托管的 runtime，
重新启动应用以重新加载 sidecar 和原生 DLL，避免旧模块缓存继续报错。

## 与主体包的关系

主体包只包含 Python、`websockets`、后端和 manifest，不包含两种 PyTorch
轮子或模型权重。这样可以让界面和本地 WebSocket 先可用，再由用户按硬件
情况选择推理层级。发布包中的主体运行时应满足：

- `local_service_ready = true`；
- 没有 CPU/CUDA PyTorch 时 `inference_ready = false`；
- 不把“无 PyTorch”误报成 WebSocket 不可用；
- 不把 CPU 版和 CUDA 版同时列为必装。

## 验证方式

后端契约测试覆盖 CPU、CUDA、未安装和导入失败四种分支；完整验证命令见
[构建与开发](BUILD.md) 的「测试与质量门禁」。
