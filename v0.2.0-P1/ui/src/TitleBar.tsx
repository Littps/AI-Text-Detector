// Custom window chrome for the Tauri shell: a glass drag strip with
// Windows-style controls. Browser dev renders nothing (no window there).
//   - decorations:false in tauri.conf.json removes the native frame
//   - data-tauri-drag-region gives the strip native drag + dblclick-maximize
//   - controls use @tauri-apps/api/window (permissions in capabilities/)
// The strip is the only surface real content scrolls beneath (body scroll
// passes under it), so its glass has something to actually refract.
import { getCurrentWindow } from "@tauri-apps/api/window";

const GlyphMinimize = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
    <path d="M0 5h10" stroke="currentColor" strokeWidth="1" />
  </svg>
);

const GlyphMaximize = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
    <rect x="0.5" y="0.5" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="1" />
  </svg>
);

const GlyphClose = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
    <path d="M0 0l10 10M10 0L0 10" stroke="currentColor" strokeWidth="1" />
  </svg>
);

export function TitleBar() {
  if (!("__TAURI_INTERNALS__" in window)) return null;
  const win = getCurrentWindow();
  return (
    <div className="titlebar glass" data-tauri-drag-region="true">
      <span className="tb-title" data-tauri-drag-region="true">
        AI Text Detector
      </span>
      <span className="tb-credit" data-tauri-drag-region="true">
        （@Github Littps/AI-Text-Detector）
      </span>
      <div className="win-controls">
        <button aria-label="最小化" onClick={() => void win.minimize()}>
          <GlyphMinimize />
        </button>
        <button aria-label="最大化 / 还原" onClick={() => void win.toggleMaximize()}>
          <GlyphMaximize />
        </button>
        <button className="wc-close" aria-label="关闭" onClick={() => void win.close()}>
          <GlyphClose />
        </button>
      </div>
    </div>
  );
}
