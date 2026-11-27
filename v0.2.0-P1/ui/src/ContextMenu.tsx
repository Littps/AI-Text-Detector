// Custom right-click menu (2026-11-21 user request): the WebView2 native menu
// is a Win32 popup that cannot follow the app's rounded, warm-paper design, so
// the contextmenu event is intercepted and this component renders instead.
//
// Behaviour: pointer-anchored (flips at the viewport edges), spring entrance
// and exit, arrow-key navigation, Esc / outside click / wheel dismissal.
// "剪切/复制/粘贴" are enabled from the actual target state, not assumed.
import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { SPRING_CLOSE } from "./spring";

export interface MenuItem {
  id: string;
  label: string;
  shortcut?: string;
  disabled?: boolean;
  /** hairline above this row (the spec's 剪切/复制/粘贴 ─ 全选 rule) */
  divider?: boolean;
  run: () => void | Promise<void>;
}

interface Props {
  x: number;
  y: number;
  items: MenuItem[];
  onClose: () => void;
}

export function ContextMenu({ x, y, items, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const closing = useRef(false);
  const [active, setActive] = useState(-1);
  const [pos, setPos] = useState<{ left: number; top: number; flipX: boolean; flipY: boolean }>({
    left: x,
    top: y,
    flipX: false,
    flipY: false,
  });

  // edge clamp runs before paint so the first frame is never off-screen
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const margin = 8;
    const { offsetWidth: w, offsetHeight: h } = el;
    const flipX = x + w + margin > window.innerWidth;
    const flipY = y + h + margin > window.innerHeight;
    setPos({
      left: flipX ? Math.max(margin, x - w) : x,
      top: flipY ? Math.max(margin, y - h) : y,
      flipX,
      flipY,
    });
  }, [x, y]);

  const requestClose = useCallback(() => {
    if (closing.current) return;
    closing.current = true;
    const el = ref.current;
    if (!el || typeof el.animate !== "function") {
      onClose();
      return;
    }
    const anim = el.animate(
      [
        { opacity: 1, transform: "scale(1)" },
        { opacity: 0, transform: "scale(0.97)" },
      ],
      { duration: Math.min(SPRING_CLOSE.duration, 240), easing: SPRING_CLOSE.easing, fill: "forwards" },
    );
    anim.onfinish = onClose;
  }, [onClose]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) requestClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        requestClose();
        return;
      }
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const step = e.key === "ArrowDown" ? 1 : -1;
        setActive((i) => {
          const n = items.length;
          let next = i;
          for (let k = 0; k < n; k++) {
            next = (next + step + n) % n;
            if (!items[next].disabled) return next;
          }
          return i;
        });
        return;
      }
      if (e.key === "Enter" && active >= 0 && !items[active]?.disabled) {
        e.preventDefault();
        void items[active].run();
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
  }, [active, items, requestClose]);

  const style: CSSProperties = {
    left: pos.left,
    top: pos.top,
    transformOrigin: `${pos.flipY ? "bottom" : "top"} ${pos.flipX ? "right" : "left"}`,
  };

  return (
    <div ref={ref} className="ctx-menu" style={style} role="menu" aria-label="文本操作">
      {items.map((item, i) => (
        <Fragment key={item.id}>
          {item.divider && <div className="ctx-sep" role="separator" />}
          <button
            type="button"
            role="menuitem"
            className={"ctx-item" + (active === i ? " active" : "")}
            disabled={item.disabled}
            onMouseEnter={() => setActive(i)}
            onClick={() => {
              void item.run();
              requestClose();
            }}
          >
            <span className="ctx-label">{item.label}</span>
            {item.shortcut && <span className="ctx-kbd">{item.shortcut}</span>}
          </button>
        </Fragment>
      ))}
    </div>
  );
}
