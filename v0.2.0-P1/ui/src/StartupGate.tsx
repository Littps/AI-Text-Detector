/**
 * First-screen gate (round-19).
 *
 * `work` used to render nothing until the first catalogue reply landed, which
 * on a slow/cold sidecar is an empty canvas - indistinguishable from the white
 * screen users report. The gate names the state, counts the wait, and after
 * `SLOW_AFTER` seconds turns into a recovery panel instead of an endless
 * spinner.
 */

export const SLOW_AFTER = 12;

const CONN_TEXT: Record<"idle" | "connecting" | "open" | "closed", string> = {
  idle: "还没有发起连接",
  connecting: "正在连接本地检测服务",
  open: "已连接，正在读取引擎清单",
  closed: "连接已断开，正在自动重连",
};

interface StartupGateProps {
  conn: "idle" | "connecting" | "open" | "closed";
  seconds: number;
  error: string | null;
  onRetry: () => void;
  onReload: () => void;
}

export function StartupGate({ conn, seconds, error, onRetry, onReload }: StartupGateProps) {
  const slow = seconds >= SLOW_AFTER;
  return (
    <section className={"startup-gate" + (slow ? " slow" : "")} aria-live="polite">
      {!slow && <div className="gate-spinner" aria-hidden="true" />}
      <h2>{slow ? "界面还没有连上检测服务" : "正在准备检测台…"}</h2>
      <p className="gate-line">
        {CONN_TEXT[conn]} · 已等待 {seconds} 秒
      </p>
      {error && <pre className="gate-error">{error}</pre>}
      {slow ? (
        <>
          <p className="hint">
            后端是独立进程，启动时要加载 torch 与模型清单。超过 {SLOW_AFTER} 秒仍无响应，
            先重试连接；仍不行就重载一次界面。
          </p>
          <div className="gate-actions">
            <button onClick={onRetry}>重试连接</button>
            <button className="primary" onClick={onReload}>
              重新加载界面
            </button>
          </div>
          <p className="hint">
            若整个窗口发白、这些按钮也点不动：任务栏托盘图标 →「重新加载界面」，或关窗重开。
          </p>
        </>
      ) : (
        <p className="hint">首次启动要读模型清单，通常 1–3 秒。</p>
      )}
    </section>
  );
}
