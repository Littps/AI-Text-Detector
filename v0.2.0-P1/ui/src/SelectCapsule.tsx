// Capsule + anchored single-select menu (2026-11-21 results rework).
//
// The results toolbar needs two of these: "筛选：全部句子" and "模型：综合".
// The native <select> cannot carry hints or the app's spring motion, and the
// mode popover is too heavy for a two-line choice - so this is the small
// sibling: pill trigger, viewport-clamped fixed menu, spring in + out.
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import { SPRING_CLOSE } from "./spring";

export interface CapsuleOption {
  id: string;
  label: string;
  hint?: string;
}

interface Props {
  /** prefix shown inside the trigger, e.g. 筛选：/ 模型： */
  prefix: string;
  value: string;
  options: CapsuleOption[];
  onChange: (id: string) => void;
  title?: string;
  /** menu width, default 268 */
  width?: number;
}

export function SelectCapsule({ prefix, value, options, onChange, title, width = 268 }: Props) {
  const [open, setOpen] = useState(false);
  const [closing, setClosing] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const closingRef = useRef(false);
  const [place, setPlace] = useState<{ left: number; dir: "below" | "above"; maxH: number }>({
    left: 0,
    dir: "below",
    maxH: 320,
  });

  const current = options.find((o) => o.id === value) ?? options[0];

  useLayoutEffect(() => {
    if (!open) return;
    const el = menuRef.current;
    const b = btnRef.current?.getBoundingClientRect();
    if (!el || !b) return;
    const h = el.offsetHeight;
    const above = b.top;
    const below = window.innerHeight - b.bottom;
    const dir: "below" | "above" = below >= h + 12 || below >= above ? "below" : "above";
    const left = Math.max(8, Math.min(b.left, window.innerWidth - width - 8));
    const maxH = Math.min(window.innerHeight - 16, Math.max(120, (dir === "below" ? below : above) - 12));
    setPlace({ left, dir, maxH });
  }, [open, width]);

  const requestClose = useCallback(() => {
    if (closingRef.current) return;
    closingRef.current = true;
    const el = menuRef.current;
    if (!el || typeof el.animate !== "function") {
      setOpen(false);
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
      setOpen(false);
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node) && !btnRef.current?.contains(e.target as Node)) {
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
  }, [open, requestClose]);

  const style: CSSProperties =
    place.dir === "below"
      ? {
          left: place.left,
          top: (btnRef.current?.getBoundingClientRect().bottom ?? 0) + 8,
          maxHeight: place.maxH,
          width,
        }
      : {
          left: place.left,
          bottom: window.innerHeight - (btnRef.current?.getBoundingClientRect().top ?? 0) + 8,
          maxHeight: place.maxH,
          width,
        };

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        className={"sel-capsule" + (open ? " open" : "")}
        title={title ?? current?.label}
        onClick={() => (open ? requestClose() : setOpen(true))}
      >
        <span className="sel-prefix">{prefix}</span>
        <span className="sel-value">{current?.label ?? "—"}</span>
        <span className="mode-caret">▾</span>
      </button>
      {open && (
        <div
          ref={menuRef}
          className={"pop-menu " + place.dir + (closing ? " closing" : "")}
          style={{ ...style, transformOrigin: place.dir === "below" ? "top left" : "bottom left" }}
          role="listbox"
        >
          {options.map((o, i) => (
            <button
              key={o.id}
              type="button"
              role="option"
              aria-selected={o.id === value}
              className={"pop-item" + (o.id === value ? " active" : "")}
              style={{ animationDelay: `${Math.min(i, 8) * 18}ms` }}
              onClick={() => {
                onChange(o.id);
                requestClose();
              }}
            >
              <span className="pop-item-body">
                <span className="pop-item-label">{o.label}</span>
                {o.hint && <span className="pop-item-hint">{o.hint}</span>}
              </span>
              {o.id === value && <span className="pop-item-check">✓</span>}
            </button>
          ))}
        </div>
      )}
    </>
  );
}
