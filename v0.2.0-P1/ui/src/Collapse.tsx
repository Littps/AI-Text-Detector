// Height collapse driven by a sampled spring (2026-11-21).
//
// `grid-template-rows: 0fr -> 1fr` lets the browser interpolate the height
// without measuring anything, and because the easing is a real spring curve
// (with a small overshoot) the panel gains a bit of mass instead of snapping.
// Content stays mounted while the exit animation plays out, then unmounts
// when requested. This keeps the spring transition while preventing a closed
// sentence list from retaining thousands of DOM nodes.
import { useEffect, useState, type ReactNode } from "react";

interface Props {
  open: boolean;
  children?: ReactNode;
  /** Optional render callback; only called while the content is mounted. */
  render?: () => ReactNode;
  className?: string;
  /** Unmount large content after the close spring instead of keeping it hot. */
  unmountOnClose?: boolean;
  /** Keep this slightly longer than the CSS collapse duration. */
  exitMs?: number;
}

export function Collapse({
  open,
  children,
  render,
  className,
  unmountOnClose = false,
  exitMs = 720,
}: Props) {
  const [mounted, setMounted] = useState(open);

  useEffect(() => {
    if (open) {
      // Make the "has been opened" flag durable before a later close. The
      // timeout keeps React's effect lint happy and is harmless because the
      // open render already shows the content synchronously.
      const timer = window.setTimeout(() => setMounted(true), 0);
      return () => window.clearTimeout(timer);
    }
    if (!unmountOnClose) return;
    const timer = window.setTimeout(() => setMounted(false), exitMs);
    return () => window.clearTimeout(timer);
  }, [open, unmountOnClose, exitMs]);

  // `open` mounts synchronously on the opening render; `mounted` only keeps
  // the old children alive while the close spring is still running.
  const content = open || mounted ? (render ? render() : children) : null;

  return (
    <div
      className={"collapse" + (open ? " open" : "") + (className ? " " + className : "")}
      aria-hidden={!open}
      /* closed content leaves the tab order even during its exit window */
      inert={!open}
    >
      <div className="collapse-inner">{content}</div>
    </div>
  );
}
