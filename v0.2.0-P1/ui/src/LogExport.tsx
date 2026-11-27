// round-30 dev option: export the sidecar log for a chosen time range.
//
// Flow: pick a range in the dialog -> sidecar concatenates its daily files and
// answers the text -> the shell's save dialog picks a path -> a tiny Rust
// command writes the bytes. In a plain browser (verification / future web
// build) there is no save dialog, so the same text goes through a Blob
// download instead of failing silently.
import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { inTauri } from "./filepicker";
import type { LogRange, LogExportPayload } from "./types";
import type { SidecarClient } from "./ws";
import type { NoticeItem } from "./Notice";

const RANGES: { days: LogRange; label: string; hint: string }[] = [
  { days: 1, label: "今天", hint: "只看今天的运行记录" },
  { days: 3, label: "近 3 天", hint: "含今天在内的三天" },
  { days: 7, label: "近 7 天", hint: "含今天在内的七天" },
];

interface Props {
  /** read lazily: the client is swapped on reconnect / HMR */
  getClient: () => SidecarClient | null;
  onNotice: (n: Omit<NoticeItem, "id">) => void;
}

export function LogExport({ getClient, onNotice }: Props) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<LogRange | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const run = async (days: LogRange) => {
    const c = getClient();
    if (!c) {
      setErr("还没有连上后端，先在「连接设置」里确认连接状态。");
      return;
    }
    setBusy(days);
    setErr(null);
    try {
      const res = await c.request<LogExportPayload>("logs/export", { days });
      const name = `ai-text-detector-logs-${res.from}_${res.to}.txt`;
      let saved: string | null = null;
      if (inTauri()) {
        const path = await save({
          defaultPath: name,
          title: "保存日志",
          filters: [{ name: "日志文本", extensions: ["txt", "log"] }],
        });
        if (!path) {
          // the user dismissed the save dialog: not an error, just stop
          setBusy(null);
          return;
        }
        await invoke("save_text_file", { path, contents: res.text });
        saved = path;
      } else {
        downloadInBrowser(name, res.text);
      }
      setOpen(false);
      onNotice({
        kind: "ok",
        title: "日志已导出",
        body: saved
          ? `${saved}（${res.files} 个文件 / ${res.lines} 行）`
          : `${name}（${res.files} 个文件 / ${res.lines} 行，浏览器下载）`,
      });
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <>
      <div className="row">
        <button onClick={() => setOpen(true)}>导出日志…</button>
        <span className="hint">按天归档在 sidecar 的 logs/ 目录，可选今天 / 近 3 天 / 近 7 天。</span>
      </div>

      {open && (
        <div className="modal-backdrop">
          <div className="modal log-export-modal">
            <h3>导出日志</h3>
            <p className="hint">
              选择要导出的时间范围。日志只记录连接、引擎加载、下载与驻留等事件，
              <strong>不包含任何待检测原文</strong>，可以直接发给开发者排查问题。
            </p>
            <div className="log-range-list">
              {RANGES.map((r) => (
                <button
                  key={r.days}
                  className="log-range"
                  disabled={busy !== null}
                  onClick={() => void run(r.days)}
                >
                  <span className="log-range-label">{r.label}</span>
                  <span className="log-range-hint">{r.hint}</span>
                  {busy === r.days && <span className="log-range-busy">导出中…</span>}
                </button>
              ))}
            </div>
            {err && <p className="warn">{err}</p>}
            <div className="modal-actions">
              <button onClick={() => setOpen(false)} disabled={busy !== null}>
                取消
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function downloadInBrowser(name: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // give the click a tick before the object URL is revoked
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
