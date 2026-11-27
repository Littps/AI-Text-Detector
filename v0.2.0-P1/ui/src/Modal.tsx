// Shared modal shell. Callers keep their content state mounted for the close
// animation and pass `closing=true`; this component has no local presence state
// and therefore cannot accidentally freeze stale children.
import { useEffect, useRef, type ReactNode } from "react";

interface Props {
  open: boolean;
  closing?: boolean;
  onClose: () => void;
  children: ReactNode;
  className?: string;
  ariaLabel?: string;
}

export function Modal({
  open,
  closing = false,
  onClose,
  children,
  className,
  ariaLabel,
}: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    dialogRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose, open]);

  if (!open) return null;
  return (
    <div
      className={"modal-backdrop" + (closing ? " closing" : "")}
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        className={"modal glass" + (className ? ` ${className}` : "") + (closing ? " closing" : "")}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel}
        tabIndex={-1}
      >
        {children}
      </div>
    </div>
  );
}
