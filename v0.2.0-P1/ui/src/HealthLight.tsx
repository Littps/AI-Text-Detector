// D22/D28 health light: six states with a human-readable reason on hover.
export type HealthState = "booting" | "downloading" | "ready" | "busy" | "stale" | "error";

const STATE_META: Record<HealthState, { color: string; label: string }> = {
  booting: { color: "#e0b34c", label: "启动中" },
  downloading: { color: "#4c8de0", label: "下载/转换中" },
  ready: { color: "#4caf72", label: "就绪" },
  busy: { color: "#9c6ade", label: "推理中" },
  stale: { color: "#c9a15a", label: "结果已过期（文本已改动）" },
  error: { color: "#d95f5f", label: "异常" },
};

interface Props {
  state: HealthState;
  reason?: string;
}

export function HealthLight({ state, reason }: Props) {
  const meta = STATE_META[state];
  return (
    <span className="health-light" title={reason ?? meta.label}>
      <span className="health-dot" style={{ background: meta.color }} />
      <span>{meta.label}</span>
    </span>
  );
}

/** Derive the health state from app facts (D28 ordering: error > busy > ...). */
export function deriveHealth(args: {
  conn: "connecting" | "open" | "closed" | "idle";
  downloading: boolean;
  casting: boolean;
  detecting: boolean;
  error: string | null;
  textDirty: boolean;
  hasResult: boolean;
}): { state: HealthState; reason?: string } {
  if (args.error) return { state: "error", reason: args.error };
  if (args.conn === "closed") return { state: "error", reason: "与 sidecar 的连接断开" };
  if (args.detecting) return { state: "busy", reason: "引擎推理中，结果将逐段流出" };
  if (args.downloading || args.casting)
    return { state: "downloading", reason: args.casting ? "模型精度转换中" : "权重下载中" };
  if (args.conn === "connecting") return { state: "booting", reason: "正在连接推理后端" };
  if (args.conn === "idle") return { state: "booting", reason: "未连接（在 Tauri 壳内会自动连接）" };
  if (args.hasResult && args.textDirty)
    return { state: "stale", reason: "文本已改动，结果基于旧文本——重新检测以更新" };
  return { state: "ready" };
}
