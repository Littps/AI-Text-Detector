// Device brief for the left rail (round-14 user request): CPU / DRAM / GPU,
// with the VRAM total the process can actually request (7.93 GB on an 8GB
// card). Every value comes from the sidecar's boot-time probe
// (backend/hardware.py) — the panel never re-queries the system, so it can
// never disagree with the budget/tier the engine loader is using.
import { fmtMb } from "./ws";
import type { HealthPayload } from "./types";

interface Props {
  hw: HealthPayload["hw"] | undefined;
}

export function DevicePanel({ hw }: Props) {
  if (!hw) {
    return (
      <div className="device-block">
        <div className="device-head">设备</div>
        <p className="device-idle">硬件档案未加载</p>
      </div>
    );
  }
  const cpu = hw.cpu_name ?? "未知 CPU";
  const threads = hw.cpu_threads ?? 0;
  const gpu = hw.gpu_name ?? "无独立显卡";
  return (
    <div className="device-block">
      <div className="device-head">设备</div>
      <dl className="device-list">
        <div className="device-row">
          <dt>CPU</dt>
          <dd>
            <span className="device-name" title={cpu}>
              {cpu}
            </span>
            {threads > 0 && <span className="device-sub">{threads} 线程</span>}
          </dd>
        </div>
        <div className="device-row">
          <dt>内存</dt>
          <dd>{hw.ram_total_mb > 0 ? fmtMb(hw.ram_total_mb) : "—"}</dd>
        </div>
        <div className="device-row">
          <dt>显卡</dt>
          <dd>
            <span className="device-name" title={gpu}>
              {gpu}
            </span>
          </dd>
        </div>
        <div className="device-row">
          <dt>显存</dt>
          <dd>{hw.vram_total_mb > 0 ? fmtMb(hw.vram_total_mb) : "—"}</dd>
        </div>
      </dl>
    </div>
  );
}
