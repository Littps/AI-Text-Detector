// round-30: bottom-right notice stack (user spec for "unloaded" feedback).
//
// Differs from the existing centre-bottom `.toast` on purpose: the toast is a
// 3.6 s one-liner for transient mistakes, while these carry a result the user
// asked for (a model left the GPU) and stay 15 s. Slide in from the right,
// slide back out, a hairline countdown so the remaining time is visible, and
// an × in the card's own top-right corner for an immediate close.
import { useCallback, useEffect, useRef, useState } from "react";

/** user spec: auto-close after 15 s */
export const NOTICE_TTL_MS = 15_000;
/** must match the .notice.leaving animation (210 ms) */
const SLIDE_MS = 210;

export type NoticeKind = "ok" | "warn" | "error";

export interface NoticeItem {
  id: number;
  title: string;
  body?: string;
  kind: NoticeKind;
  /** ms; 0 keeps it until closed. Defaults to NOTICE_TTL_MS. */
  ttl?: number;
}

export function Notices({
  items,
  onDismiss,
}: {
  items: NoticeItem[];
  onDismiss: (id: number) => void;
}) {
  if (items.length === 0) return null;
  return (
    <div className="notice-stack" role="status" aria-live="polite">
      {items.map((n) => (
        <NoticeCard key={n.id} item={n} onDismiss={onDismiss} />
      ))}
    </div>
  );
}

function NoticeCard({
  item,
  onDismiss,
}: {
  item: NoticeItem;
  onDismiss: (id: number) => void;
}) {
  const [leaving, setLeaving] = useState(false);
  const closed = useRef(false);

  const close = useCallback(() => {
    if (closed.current) return;
    closed.current = true;
    setLeaving(true);
    // let the slide-out finish before React drops the node
    window.setTimeout(() => onDismiss(item.id), SLIDE_MS);
  }, [item.id, onDismiss]);

  const ttl = item.ttl ?? NOTICE_TTL_MS;
  useEffect(() => {
    if (ttl <= 0) return;
    const t = window.setTimeout(close, ttl);
    return () => window.clearTimeout(t);
  }, [close, ttl]);

  return (
    <div className={`notice notice-${item.kind}${leaving ? " leaving" : ""}`}>
      <div className="notice-main">
        <div className="notice-title">{item.title}</div>
        {item.body && <div className="notice-text">{item.body}</div>}
      </div>
      <button type="button" className="notice-x" onClick={close} aria-label="关闭提示" title="关闭">
        <svg viewBox="0 0 12 12" width="10" height="10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
          <path d="M2.5 2.5 9.5 9.5M9.5 2.5 2.5 9.5" />
        </svg>
      </button>
      {ttl > 0 && <span className="notice-bar" style={{ animationDuration: `${ttl}ms` }} />}
    </div>
  );
}
