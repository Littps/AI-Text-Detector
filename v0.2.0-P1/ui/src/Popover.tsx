// Anchored popover primitive (2026-11-21): one shared shell for the two new
// click-to-inspect surfaces (engine list in the composer strip, per-sentence
// model breakdown). Same mechanics as SelectCapsule - fixed positioning,
// viewport flip + clamp, spring entrance, WAAPI spring exit that plays out
// before unmounting - but without the "single select" semantics.
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { SPRING_CLOSE } from "./spring";

interface Props {
  open: boolean;
  /** element the panel is anchored to (under it, or above when tight) */
  anchor: HTMLElement | null;
  /** panel width in px, default 260 */
  width?: number;
  onClose: () => void;
  children: ReactNode;
  ariaLabel?: string;
}

export function Popover({ open, anchor, width = 260, onClose, children, ariaLabel }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const closingRef = useRef(false);
  const [closing, setClosing] = useState(false);
  const [place, setPlace] = useState<{ left: number; top: number; dir: "below" | "above"; maxH: number }>({
    left: 0,
    top: 0,
    dir: "below",
    maxH: 320,
  });

  useLayoutEffect(() => {
    if (!open || !anchor) return;
    const el = ref.current;
    if (!el) return;
    const b = anchor.getBoundingClientRect();
    // scrollHeight, not offsetHeight: the panel is already capped by maxHeight
    const h = Math.min(el.scrollHeight, 320);
    const above = b.top;
    const below = window.innerHeight - b.bottom;
    const dir: "below" | "above" = below >= h + 12 || below >= above ? "below" : "above";
    const left = Math.max(8, Math.min(b.left, window.innerWidth - width - 8));
    const top = dir === "below" ? b.bottom + 8 : b.top - h - 8;
    const maxH = Math.min(window.innerHeight - 16, Math.max(120, (dir === "below" ? below : above) - 12));
    setPlace({ left, top, dir, maxH });
  }, [open, anchor, width]);

  const requestClose = useCallback(() => {
    if (closingRef.current) return;
    closingRef.current = true;
    const el = ref.current;
    if (!el || typeof el.animate !== "function") {
      closingRef.current = false;
      onClose();
      return;
    }
    setClosing(true);
    const anim = el.animate(
      [
        { opacity: 1, transform: "scale(1) translateY(0)" },
        { opacity: 0, transform: "scale(0.97) translateY(3px)" },
      ],
      { duration: SPRING_CLOSE.duration, easing: SPRING_CLOSE.easing, fill: "forwards" },
    );
    anim.onfinish = () => {
      closingRef.current = false;
      setClosing(false);
      onClose();
    };
  }, [onClose]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node) && !anchor?.contains(e.target as Node)) {
        requestClose();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        requestClose();
      }
    };
    window.addEventListener("mousedown", onDown, true);
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("wheel", requestClose, { passive: true });
    window.addEventListener("resize", requestClose);
    return () => {
      window.removeEventListener("mousedown", onDown, true);
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("wheel", requestClose);
      window.removeEventListener("resize", requestClose);
    };
  }, [open, anchor, requestClose]);

  if (!open) return null;
  return (
    <div
      ref={ref}
      className={"pop-menu " + place.dir + (closing ? " closing" : "")}
      style={{
        left: place.left,
        top: place.top,
        width,
        maxHeight: place.maxH,
        transformOrigin: place.dir === "below" ? "top left" : "bottom left",
      }}
      role="dialog"
      aria-label={ariaLabel}
    >
      {children}
    </div>
  );
}
