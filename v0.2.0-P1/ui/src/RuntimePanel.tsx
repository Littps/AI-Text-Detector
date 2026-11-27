import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { writeClipboardText } from "./clipboard";
import { inTauri } from "./filepicker";
import type {
  DlStatusPayload,
  DownloadSource,
  RuntimeComponent,
  RuntimeStatusPayload,
} from "./types";
import { fmtBytes } from "./ws";

interface Props {
  status: RuntimeStatusPayload | null;
  dl: DlStatusPayload | null;
  downloadSource: DownloadSource;
  onRefresh: () => Promise<void>;
  onOpenModels: () => void;
  onError: (message: string) => void;
}

type CopyState = "idle" | "ok" | "ok-out" | "fail" | "fail-out";
type GuideTask = {
  id: string;
  label: string;
  reason: string;
  officialUrl: string;
  mirrorUrl?: string;
  kind: "external" | "models";
  required: boolean;
};

const MODE_LABEL: Record<string, string> = {
  bundled: "应用托管 runtime",
  development: "开发 venv",
  external: "外部 Python",
};

const DOWNLOAD_LINKS: Record<string, string> = {
  github: "https://github.com/Littps/AI-Text-Detector/releases/latest",
  python: "https://www.python.org/downloads/windows/",
  vc: "https://aka.ms/vs/17/release/vc_redist.x64.exe",
  torch: "https://pytorch.org/get-started/locally/",
  torchCpu: "https://download.pytorch.org/whl/cpu",
  torchCpuMirror: "https://pypi.tuna.tsinghua.edu.cn/simple/torch/",
  torchMirror: "https://pypi.tuna.tsinghua.edu.cn/simple/torch/",
  transformers: "https://pypi.org/project/transformers/",
  transformersMirror: "https://pypi.tuna.tsinghua.edu.cn/simple/transformers/",
  tokenizers: "https://pypi.org/project/tokenizers/",
  tokenizersMirror: "https://pypi.tuna.tsinghua.edu.cn/simple/tokenizers/",
  safetensors: "https://pypi.org/project/safetensors/",
  safetensorsMirror: "https://pypi.tuna.tsinghua.edu.cn/simple/safetensors/",
  numpy: "https://pypi.org/project/numpy/",
  numpyMirror: "https://pypi.tuna.tsinghua.edu.cn/simple/numpy/",
  accelerate: "https://pypi.org/project/accelerate/",
  accelerateMirror: "https://pypi.tuna.tsinghua.edu.cn/simple/accelerate/",
  pypdf: "https://pypi.org/project/pypdf/",
  pypdfMirror: "https://pypi.tuna.tsinghua.edu.cn/simple/pypdf/",
  nvidia: "https://www.nvidia.cn/geforce/drivers/",
} as const;

function stateLabel(component: RuntimeComponent): string {
  if (component.state === "ready") return "正常";
  if (component.state === "warning") return "需注意";
  return "缺失";
}

export function RuntimePanel({
  status,
  dl,
  downloadSource,
  onRefresh,
  onOpenModels,
  onError,
}: Props) {
  const [busy, setBusy] = useState(false);
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const copyTimers = useRef<number[]>([]);

  const clearCopyTimers = () => {
    copyTimers.current.forEach((timer) => window.clearTimeout(timer));
    copyTimers.current = [];
  };

  useEffect(
    () => () => {
      copyTimers.current.forEach((timer) => window.clearTimeout(timer));
    },
    [],
  );

  const refresh = async () => {
    setBusy(true);
    try {
      await onRefresh();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const copyDiagnostic = async () => {
    if (!status) return;
    const text = JSON.stringify(
      {
        generated_at: status.generated_at,
        mode: status.mode,
        python: status.python_version,
        python_executable: status.python_executable,
        runtime_root: status.runtime_root,
        site_packages: status.site_packages,
        recommended_wheel: status.recommended_wheel,
        local_service_ready: status.local_service_ready,
        inference_ready: status.inference_ready,
        components: status.components,
        dlls: status.dlls,
        notes: status.notes,
      },
      null,
      2,
    );
    const ok = await writeClipboardText(text);
    clearCopyTimers();
    setCopyState(ok ? "ok" : "fail");
    const hold = ok ? 1600 : 3200;
    copyTimers.current.push(
      window.setTimeout(() => setCopyState(ok ? "ok-out" : "fail-out"), hold),
      window.setTimeout(() => setCopyState("idle"), hold + 160),
    );
  };

  const openUrl = async (url: string) => {
    if (inTauri()) {
      await invoke("open_external_url", { url });
    } else {
      window.open(url, "_blank", "noopener,noreferrer");
    }
  };

  if (!status) {
    return (
      <section className="settings-section runtime-loading" aria-live="polite">
        <h2>基础服务检查中</h2>
        <p className="hint">正在读取 Python、WebSocket、PyTorch 与系统运行库状态。</p>
      </section>
    );
  }

  const tasks: GuideTask[] = [];
  for (const component of status.components) {
    if (component.state === "ready" || component.id === "vc-runtime") continue;
    const isBase =
      component.id === "python" || component.id === "websockets";
    let officialUrl = DOWNLOAD_LINKS.github;
    let mirrorUrl: string | undefined;
    if (component.id === "torch") {
      officialUrl = DOWNLOAD_LINKS.torch;
      if (status.recommended_wheel !== "cu129") mirrorUrl = DOWNLOAD_LINKS.torchMirror;
    } else if (component.id === "torch-cpu") {
      officialUrl = DOWNLOAD_LINKS.torchCpu;
      mirrorUrl = DOWNLOAD_LINKS.torchCpuMirror;
    } else if (component.id === "transformers") {
      officialUrl = DOWNLOAD_LINKS.transformers;
      mirrorUrl = DOWNLOAD_LINKS.transformersMirror;
    } else if (component.id === "tokenizers") {
      officialUrl = DOWNLOAD_LINKS.tokenizers;
      mirrorUrl = DOWNLOAD_LINKS.tokenizersMirror;
    } else if (component.id === "safetensors") {
      officialUrl = DOWNLOAD_LINKS.safetensors;
      mirrorUrl = DOWNLOAD_LINKS.safetensorsMirror;
    } else if (component.id === "numpy") {
      officialUrl = DOWNLOAD_LINKS.numpy;
      mirrorUrl = DOWNLOAD_LINKS.numpyMirror;
    } else if (component.id === "accelerate") {
      officialUrl = DOWNLOAD_LINKS.accelerate;
      mirrorUrl = DOWNLOAD_LINKS.accelerateMirror;
    } else if (component.id === "pypdf") {
      officialUrl = DOWNLOAD_LINKS.pypdf;
      mirrorUrl = DOWNLOAD_LINKS.pypdfMirror;
    } else if (component.id === "python") {
      officialUrl = DOWNLOAD_LINKS.python;
    }
    tasks.push({
      id: component.id,
      label: component.label,
      reason: component.detail,
      officialUrl,
      mirrorUrl,
      kind: "external",
      required: isBase || component.required,
    });
  }

  const vcMissing = status.components.find(
    (component) => component.id === "vc-runtime" && component.state !== "ready",
  );
  if (vcMissing) {
    tasks.push({
      id: "vc-runtime",
      label: vcMissing.label,
      reason: vcMissing.detail,
      officialUrl: DOWNLOAD_LINKS.vc,
      kind: "external",
      required: true,
    });
  }

  const driverUnknown =
    Boolean(status.hardware.gpu_name) &&
    (!status.hardware.driver_version ||
      (status.recommended_wheel === "cu129" && !status.hardware.cuda));
  if (driverUnknown) {
    tasks.push({
      id: "nvidia-driver",
      label: "NVIDIA 显卡驱动",
      reason: status.hardware.driver_version
        ? `检测到 ${status.hardware.gpu_name}，但当前驱动未通过 cu129 检查`
        : `检测到 ${status.hardware.gpu_name}，但无法读取驱动版本`,
      officialUrl: DOWNLOAD_LINKS.nvidia,
      kind: "external",
      required: status.recommended_wheel === "cu129",
    });
  }

  const missingDefaultModels = Object.entries(dl?.engines ?? {})
    .filter(([, item]) => item.default_selected && !item.installed)
    .map(([engineId]) => engineId);
  if (missingDefaultModels.length > 0) {
    tasks.push({
      id: "model-weights",
      label: "推荐模型权重",
      reason: `${missingDefaultModels.length} 个默认模型尚未下载：${missingDefaultModels.join("、")}`,
      officialUrl: DOWNLOAD_LINKS.github,
      kind: "models",
      required: true,
    });
  }

  const externalTasks = tasks.filter((task) => task.kind === "external");
  const selectedTasks = externalTasks.filter((task) => selected.has(task.id));
  const chooseUrl = (task: GuideTask) => {
    if (downloadSource === "mirror" && task.mirrorUrl) {
      return {
        url: task.mirrorUrl,
        label: task.id === "torch-cpu" ? "CPU wheel 镜像" : "镜像下载",
      };
    }
    return {
      url: task.officialUrl,
      label:
        task.id === "torch-cpu"
          ? "CPU wheel"
          : task.id === "nvidia-driver"
            ? "官方驱动页"
            : "官方下载",
    };
  };
  const toggleTask = (taskId: string) => {
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(taskId)) next.delete(taskId);
      else next.add(taskId);
      return next;
    });
  };
  const openSelected = async () => {
    try {
      for (const task of selectedTasks) {
        await openUrl(chooseUrl(task).url);
      }
    } catch (e) {
      onError(String(e));
    }
  };

  const copyKind = copyState.startsWith("ok")
    ? "ok"
    : copyState.startsWith("fail")
      ? "fail"
      : null;
  const copyLeaving = copyState.endsWith("-out");
  const readyCount = status.components.filter((item) => item.state === "ready").length;

  return (
    <>
      <section className="settings-section runtime-overview" id="runtime-overview">
        <div className="runtime-title-row">
          <div>
            <h2>基础服务</h2>
            <p className="hint">
              当前来源：{MODE_LABEL[status.mode] ?? status.mode} · Python{" "}
              {status.python_version}
            </p>
          </div>
          <div className="runtime-actions">
            <button onClick={() => void refresh()} disabled={busy}>
              {busy ? "检查中…" : "重新检测"}
            </button>
            <button onClick={() => void copyDiagnostic()} aria-describedby="runtime-copy-state">
              复制诊断
            </button>
          </div>
        </div>

        <div className="runtime-readiness-grid">
          <div className={"runtime-readiness " + (status.local_service_ready ? "ready" : "missing")}>
            <span className="runtime-readiness-dot" aria-hidden="true" />
            <div>
              <strong>界面与本地服务</strong>
              <small>
                {status.local_service_ready
                  ? "WebSocket 通道可建立"
                  : "Python / WebSocket 组件不完整"}
              </small>
            </div>
            <span className="runtime-ready-label">
              {status.local_service_ready ? "已就绪" : "不可用"}
            </span>
          </div>
          <div className={"runtime-readiness " + (status.inference_ready ? "ready" : "warning")}>
            <span className="runtime-readiness-dot" aria-hidden="true" />
            <div>
              <strong>模型推理</strong>
              <small>
                {status.inference_ready
                  ? `推荐 ${status.recommended_wheel} 运行时`
                  : "核心推理组件缺失，检测暂不可用"}
              </small>
            </div>
            <span className="runtime-ready-label">
              {status.inference_ready ? "已就绪" : "未就绪"}
            </span>
          </div>
        </div>

        <div className="runtime-hardware-row">
          <span>
            GPU{" "}
            <b>{status.hardware.gpu_name ?? "未检测到独立显卡"}</b>
          </span>
          <span>
            驱动 <b>{status.hardware.driver_version ?? "—"}</b>
          </span>
          <span>
            推理目标 <b>{status.recommended_wheel}</b>
          </span>
          <span>
            当前 PyTorch{" "}
            <b>
              {status.torch_variant === "cpu"
                ? "CPU 版"
                : status.torch_variant === "cuda"
                  ? "CUDA 构建"
                  : "未就绪"}
            </b>
          </span>
          <span>
            下载源 <b>{downloadSource === "mirror" ? "镜像优先" : "官方优先"}</b>
          </span>
        </div>

        <div className="row runtime-copy-row">
          <span
            id="runtime-copy-state"
            className={
              "copy-state" +
              (copyKind ? " copy-state-" + copyKind : "") +
              (copyLeaving ? " leaving" : "")
            }
            role="status"
            aria-live="polite"
          >
            {copyKind === "ok"
              ? "诊断信息已复制"
              : copyKind === "fail"
                ? "剪贴板不可用，请手动选择下方组件路径"
                : ""}
          </span>
        </div>
      </section>

      <section className="settings-section" id="runtime-guidance">
        <div className="runtime-title-row">
          <div>
            <h2>缺失项与下载</h2>
            <p className="hint">
              页面只给出组件和官方/镜像下载入口，不自动安装驱动，也不在 sidecar
              运行中替换正在使用的原生库。
            </p>
          </div>
          <div className="runtime-actions">
            <button
              className="primary"
              onClick={() => void openSelected()}
              disabled={selectedTasks.length === 0}
            >
              打开所选下载页（{selectedTasks.length}）
            </button>
          </div>
        </div>
        {tasks.length === 0 ? (
          <div className="runtime-guidance-empty">
            <strong>当前没有需要处理的必需组件</strong>
            <span>基础服务、推理运行时和默认模型权重均已就绪。</span>
          </div>
        ) : (
          <div className="runtime-guidance-list">
            {tasks.map((task) => {
              const choice = chooseUrl(task);
              const fallback =
                task.mirrorUrl && choice.label === "官方下载"
                  ? { url: task.mirrorUrl, label: "镜像下载" }
                  : { url: task.officialUrl, label: "官方下载" };
              return (
                <div className="runtime-guide-row" key={task.id}>
                  {task.kind === "external" ? (
                    <label className="runtime-guide-check">
                      <input
                        type="checkbox"
                        checked={selected.has(task.id)}
                        onChange={() => toggleTask(task.id)}
                      />
                      <span className="runtime-guide-copy">
                        <strong>{task.label}</strong>
                        <small>{task.reason}</small>
                      </span>
                    </label>
                  ) : (
                    <span className="runtime-guide-copy">
                      <strong>{task.label}</strong>
                      <small>{task.reason}</small>
                    </span>
                  )}
                  <span className={"runtime-guide-required" + (task.required ? " required" : "")}>
                    {task.required ? "影响检测" : "可选"}
                  </span>
                  <div className="runtime-guide-actions">
                    {task.kind === "models" ? (
                      <button onClick={onOpenModels}>前往模型管理</button>
                    ) : (
                      <>
                        <button
                          onClick={() =>
                            void openUrl(choice.url).catch((e) => onError(String(e)))
                          }
                        >
                          {choice.label}
                        </button>
                        {(task.mirrorUrl || choice.label === "镜像下载") && (
                          <button
                            className="runtime-link-secondary"
                            onClick={() =>
                              void openUrl(fallback.url).catch((e) => onError(String(e)))
                            }
                          >
                            {fallback.label}
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      <section className="settings-section" id="runtime-components">
        <h2>组件状态</h2>
        <p className="hint">
          {readyCount} / {status.components.length} 项正常。启动所需和推理所需的组件分开呈现，
          避免把“界面能打开”误当成“模型一定能运行”。
        </p>
        <div className="runtime-component-grid">
          {status.components.map((item, index) => (
            <article
              className={"runtime-component-card state-" + item.state}
              key={item.id}
              style={{ animationDelay: `${index * 34}ms` }}
            >
              <div className="runtime-component-head">
                <div>
                  <span className="runtime-role">{item.role}</span>
                  <h3>{item.label}</h3>
                </div>
                <span className={"runtime-state state-" + item.state}>
                  {stateLabel(item)}
                </span>
              </div>
              <div className="runtime-version">
                {item.version ?? (item.id === "vc-runtime" ? "系统 DLL" : "未检测到")}
              </div>
              <p>{item.detail}</p>
              <div className="runtime-meta">
                <span>{item.managed_by}</span>
                <span>
                  {item.id === "torch-cpu"
                    ? item.required
                      ? "影响检测"
                      : "可选替代"
                    : item.required
                      ? "启动必需"
                    : item.role === "文档导入"
                      ? "功能依赖"
                      : "推理依赖"}
                </span>
              </div>
              {item.location && (
                <code className="runtime-path" title={item.location}>
                  {item.location}
                </code>
              )}
            </article>
          ))}
        </div>
      </section>

      <section className="settings-section" id="runtime-paths">
        <h2>运行位置</h2>
        <div className="runtime-path-list">
          <div>
            <span>应用目录</span>
            <code>{status.app_root}</code>
          </div>
          <div>
            <span>运行时目录</span>
            <code>{status.runtime_root}</code>
          </div>
          <div>
            <span>Python 包目录</span>
            <code>{status.site_packages}</code>
          </div>
          <div>
            <span>解释器</span>
            <code>{status.python_executable}</code>
          </div>
        </div>
        <h3 className="runtime-subtitle">Visual C++ DLL</h3>
        <div className="runtime-dll-list">
          {status.dlls.map((dll) => (
            <div className={"runtime-dll " + (dll.found ? "found" : "missing")} key={dll.name}>
              <span className="runtime-dll-name">{dll.name}</span>
              <span>{dll.found ? fmtBytes(dll.bytes) : "未找到"}</span>
              <code title={dll.path ?? undefined}>{dll.path ?? "—"}</code>
            </div>
          ))}
        </div>
        {status.notes.length > 0 && (
          <div className="runtime-notes">
            {status.notes.map((note) => (
              <p key={note}>{note}</p>
            ))}
          </div>
        )}
      </section>
    </>
  );
}
