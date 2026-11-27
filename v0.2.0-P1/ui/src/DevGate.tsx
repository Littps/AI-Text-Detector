// Developer-options gate (round-6 user request): two-layer confirmation
// before unlocking dev-only content. Layer 1 states what is being unlocked;
// layer 2 repeats it and force-locks the confirm button for 5 seconds so it
// cannot be clicked through mindlessly. Unlock persists per session (local-
// storage flag), with an explicit lock-up button always available.
import { useEffect, useRef, useState, type ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";
import { inTauri } from "./filepicker";

const DEV_KEY = "aigc.dev.unlocked";
const LOCK_SECONDS = 5;

// the exact line 「复制变量说明」puts on the clipboard; also rendered inline so
// a blocked clipboard still leaves something the user can select by hand
const TIER_VAR_LINE = "SIDE_FORCE_TIER=cpu|standard|plus  # 重启 sidecar 生效（D22）";

/** execCommand fallback: select a hidden textarea, copy, clean up. */
function legacyCopy(text: string): boolean {
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

export function loadDevUnlocked(): boolean {
  try {
    return localStorage.getItem(DEV_KEY) === "1";
  } catch {
    return false;
  }
}

function saveDevUnlocked(v: boolean): void {
  try {
    if (v) localStorage.setItem(DEV_KEY, "1");
    else localStorage.removeItem(DEV_KEY);
  } catch {
    /* private mode: session-only unlock */
  }
}

interface Props {
  unlocked: boolean;
  onUnlock: (v: boolean) => void;
  /** connection panel (round-6c: moved out of the settings view) */
  conn?: ReactNode;
  /** device & environment block (round-6: debug info lives here) */
  env?: ReactNode;
  /** round-30: log export (range picker + native save) */
  logs?: ReactNode;
}

export function DevOptions({ unlocked, onUnlock, conn, env, logs }: Props) {
  const [gate, setGate] = useState<1 | 2 | null>(null);
  const [lockLeft, setLockLeft] = useState(0);
  const [closing, setClosing] = useState(false);
  const timerRef = useRef<number | undefined>(undefined);

  // 5-second forced lock on layer 2; cancels cleanly on dismissal
  useEffect(() => {
    if (gate !== 2) {
      window.clearInterval(timerRef.current);
      return;
    }
    setLockLeft(LOCK_SECONDS);
    timerRef.current = window.setInterval(() => {
      setLockLeft((s) => {
        if (s <= 1) {
          window.clearInterval(timerRef.current);
          return 0;
        }
        return s - 1;
      });
    }, 1000);
    return () => window.clearInterval(timerRef.current);
  }, [gate]);

  // iOS-style exit: play the shrink+fade first, then commit the state change,
  // so the gate never vanishes abruptly mid-animation.
  const dismissWith = (action: () => void) => {
    if (closing) return;
    setClosing(true);
    window.setTimeout(() => {
      setGate(null);
      setClosing(false);
      action();
    }, 150);
  };

  const dismiss = () => dismissWith(() => {});

  const finishUnlock = () => {
    dismissWith(() => {
      saveDevUnlocked(true);
      onUnlock(true);
    });
  };

  if (unlocked) {
    return (
      <section className="settings-section dev-section" id="dev-options">
        <h2>
          开发者选项
          <span className="dev-badge">已开启</span>
        </h2>
        <p className="hint">
          以下内容面向开发调试，生产使用不需要。随时可关闭。
        </p>
        <div className="row">
          <button
            onClick={() => {
              saveDevUnlocked(false);
              onUnlock(false);
            }}
          >
            关闭开发者选项
          </button>
        </div>
        <DevContent conn={conn} env={env} logs={logs} />
      </section>
    );
  }

  return (
    <>
      <section className="settings-section" id="dev-options">
        <h2>开发者选项（默认关闭）</h2>
        <p className="hint">
          开发调试专用：协议原始帧、硬件档位强制变量等。进入需要两层确认，
          第二层含 5 秒强制锁定。
        </p>
        <div className="row">
          <button onClick={() => setGate(1)}>进入开发者选项…</button>
        </div>
      </section>

      {gate !== null && (
        <div className={"modal-backdrop gate-backdrop" + (closing ? " closing" : "")}>
          <div className="modal dev-gate-modal">
            {/* key={gate}: layer switches re-mount, replaying the rise-in */}
            <div className="gate-step" key={gate}>
              {gate === 1 ? (
                <>
                  <h3>确认进入开发者选项？</h3>
                  <p>
                    开发者选项包含<strong>面向开发调试</strong>的功能（协议原始帧、
                    连接设置、环境变量强制档位等）。这些功能<strong>不影响检测功能本身</strong>，
                    但展示的信息是给开发者看的，普通使用无需开启。
                  </p>
                  <div className="modal-actions">
                    <button onClick={dismiss}>取消</button>
                    <button className="primary" onClick={() => setGate(2)}>
                      继续
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <h3>再次确认</h3>
                  <p>
                    你正在开启<strong>开发者选项</strong>。此开关会记住在本机；
                    后续所有开发向功能都会出现在这里。确认继续？
                  </p>
                  <div className="modal-actions">
                    <button onClick={dismiss}>取消</button>
                    <button
                      className="primary"
                      disabled={lockLeft > 0}
                      onClick={finishUnlock}
                    >
                      {lockLeft > 0 ? `请阅读（${lockLeft}s）` : "确认开启"}
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/** Real dev-facing content, populated as development needs grow. */
function DevContent({
  conn,
  env,
  logs,
}: {
  conn?: ReactNode;
  env?: ReactNode;
  logs?: ReactNode;
}) {
  const [showRaw, setShowRaw] = useState(false);
  const [raw, setRaw] = useState<string>("");
  const [consoleState, setConsoleState] = useState<{
    supported: boolean;
    visible: boolean;
    preference: boolean;
  } | null>(() =>
    inTauri() ? null : { supported: false, visible: false, preference: false },
  );
  const [consoleBusy, setConsoleBusy] = useState(false);
  const [consoleError, setConsoleError] = useState<string | null>(null);
  // round-33/34: the copy button used to be silent on both paths, so a refused
  // Clipboard API looked identical to a successful copy. Track it explicitly
  // and fall back to selection+execCommand (works in WebView2 when the async
  // API is blocked by permissions). The state is 5-valued on purpose: *-out
  // exists because the text used to vanish on a hard cut — entering animated,
  // leaving instant, which read as a flicker rather than a dismissal.
  type CopyState = "idle" | "ok" | "ok-out" | "fail" | "fail-out";
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const copyTimers = useRef<number[]>([]);
  const clearCopyTimers = () => {
    copyTimers.current.forEach((t) => window.clearTimeout(t));
    copyTimers.current = [];
  };

  useEffect(
    () => () => {
      copyTimers.current.forEach((t) => window.clearTimeout(t));
    },
    [],
  );

  useEffect(() => {
    if (!inTauri()) return;
    let cancelled = false;
    const load = () => {
      void invoke<{ supported: boolean; visible: boolean; preference: boolean }>(
        "console_visibility_status",
      )
        .then((next) => {
          if (!cancelled) {
            setConsoleState(next);
            setConsoleError(null);
          }
        })
        .catch((e) => {
          if (!cancelled) setConsoleError(String(e));
        });
    };
    load();
    const timer = window.setInterval(load, 1200);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const toggleConsole = async () => {
    if (!consoleState?.supported || consoleBusy) return;
    setConsoleBusy(true);
    setConsoleError(null);
    try {
      const next = await invoke<{ supported: boolean; visible: boolean; preference: boolean }>(
        "set_console_visibility",
        { visible: !consoleState.visible },
      );
      setConsoleState(next);
    } catch (e) {
      setConsoleError(String(e));
    } finally {
      setConsoleBusy(false);
    }
  };

  const copyTier = async () => {
    const text = TIER_VAR_LINE;
    let ok = false;
    try {
      await navigator.clipboard.writeText(text);
      ok = true;
    } catch {
      ok = legacyCopy(text);
    }
    clearCopyTimers();
    setCopyState(ok ? "ok" : "fail");
    // hold long enough to read, then play the exit before emptying the slot
    const hold = ok ? 1600 : 3200;
    copyTimers.current.push(
      window.setTimeout(() => setCopyState(ok ? "ok-out" : "fail-out"), hold),
      window.setTimeout(() => setCopyState("idle"), hold + 160),
    );
  };
  const copyKind = copyState.startsWith("ok") ? "ok" : copyState.startsWith("fail") ? "fail" : null;
  const copyLeaving = copyState.endsWith("-out");

  return (
    <div className="dev-content">
      <h4>启动与调试</h4>
      <div className="dev-toggle-row">
        <div>
          <strong>显示独立调试日志窗口</strong>
          <p className="hint">
            应用自带控制台和 Python sidecar 始终无窗口启动。开启后只启动一个独立的
            只读日志窗口，关闭它不会关闭应用。开关立即生效，并在下次启动时保留。
          </p>
        </div>
        <label
          className={
            "switch-control" +
            (consoleState?.visible ? " on" : "") +
            (consoleBusy ? " busy" : "")
          }
          title={
            consoleState?.supported
              ? consoleState.visible
                ? "关闭调试日志窗口"
                : "打开调试日志窗口"
              : "当前构建没有可显示的控制台"
          }
        >
          <input
            type="checkbox"
            role="switch"
            aria-label="显示独立调试日志窗口"
            checked={consoleState?.visible ?? false}
            disabled={!consoleState?.supported || consoleBusy}
            onChange={() => void toggleConsole()}
          />
          <span className="switch-track" aria-hidden="true">
            <span className="switch-thumb" />
          </span>
        </label>
      </div>
      <p className="hint">
        {consoleState === null
          ? "正在读取调试日志窗口状态…"
          : consoleState.supported
            ? consoleState.visible
              ? "独立日志窗口已打开。关闭该窗口只会结束日志查看器。"
              : "主应用控制台保持隐藏。开启后才会显示独立日志窗口。"
            : "当前是 release 构建，不支持显示调试日志窗口。"}
      </p>
      {consoleError && <p className="hint dev-danger">{consoleError}</p>}

      {conn && (
        <>
          <h4>连接设置（调试用）</h4>
          <p className="hint">
            桌面应用会自动连接 sidecar，一般无需改动；端口 / Token 仅在手动排查连接问题时使用。
          </p>
          {conn}
        </>
      )}

      <h4>设备与环境信息（PyTorch / 档位 / 环境变量）</h4>
      <p className="hint">
        来自 sidecar health 探测：硬件档位判定依据、推理栈版本与开发用环境变量。
        SIDE_TOKEN 只报告是否设置，值永不回显。
      </p>
      {env ?? <p className="hint">（尚未连接 sidecar，暂无数据）</p>}

      <h4>硬件档位强制变量（调试用）</h4>
      <p className="hint">
        设置环境变量 <code>SIDE_FORCE_TIER</code> 为 cpu / standard / plus
        可强制硬件档位（source=forced），用于验证不同档位下的引擎选择与玻璃降级。
        修改后需重启 sidecar。
      </p>
      <div className="row">
        <button onClick={() => void copyTier()} aria-describedby="copy-tier-state">
          复制变量说明
        </button>
        <span
          id="copy-tier-state"
          className={
            "copy-state" +
            (copyKind ? " copy-state-" + copyKind : "") +
            (copyLeaving ? " leaving" : "")
          }
          role="status"
          aria-live="polite"
        >
          {copyKind === "ok"
            ? "已复制到剪贴板"
            : copyKind === "fail"
              ? "系统拒绝了剪贴板权限，请手动选中下面这行"
              : ""}
        </span>
      </div>
      <code className="copy-tier-line">{TIER_VAR_LINE}</code>

      <h4>日志导出（按时间范围）</h4>
      <p className="hint">
        sidecar 每天写一个日志文件，这里按范围拼成一份纯文本导出。
        报 bug 时把导出的文件发来即可，日志不含待检测原文。
      </p>
      {logs ?? <p className="hint">（日志模块未挂载）</p>}

      <h4>协议原始帧查看器</h4>
      <p className="hint">
        展示最近一次 health / engines 响应的原始 JSON（调试协议用）。
      </p>
      <div className="row">
        <button
          onClick={() => {
            if (showRaw) {
              setShowRaw(false);
              return;
            }
            const c = (window as unknown as { __lastFrames?: unknown }).__lastFrames;
            setRaw(JSON.stringify(c ?? "（暂无缓存帧）", null, 2));
            setShowRaw(true);
          }}
        >
          {showRaw ? "收起" : "查看原始帧"}
        </button>
      </div>
      {showRaw && <pre className="log">{raw}</pre>}
    </div>
  );
}
