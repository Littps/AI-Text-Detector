// Read-only capsule + popover (2026-11-21): the composer strip used to print
// every engine name inline, which truncated to "…" as soon as the selection
// grew. Now the count is the button and the list lives in a popover.
import { useState, type ReactNode } from "react";
import { Popover } from "./Popover";

interface Props {
  label: string;
  /** panel width, default 288 */
  width?: number;
  children: ReactNode;
}

export function InfoCapsule({ label, width = 288, children }: Props) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        className={"sel-capsule info" + (open ? " open" : "")}
        aria-expanded={open}
        onClick={(e) => {
          const el = e.currentTarget;
          if (open) {
            setOpen(false);
          } else {
            setAnchor(el);
            setOpen(true);
          }
        }}
      >
        <span className="sel-value">{label}</span>
        <span className="mode-caret">▾</span>
      </button>
      <Popover open={open} anchor={anchor} width={width} onClose={() => setOpen(false)} ariaLabel={label}>
        {children}
      </Popover>
    </>
  );
}
