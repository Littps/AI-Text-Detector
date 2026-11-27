// Attachment cards for the composer (D40).
//
// The detector sees one document, but the user may hand it several files; the
// cards therefore double as the order editor. Order is the array order in
// App state, which is exactly what detect sends as "attachments": [id].
//
// Reordering is HTML5 drag-and-drop inside the list. That works in the Tauri
// webview too: the native dragDrop handler only swallows drags that *enter the
// window with files*, while element drags stay ordinary DOM events.
import { useState } from "react";
import type { AttachmentMeta } from "./types";

const BADGE: Record<string, string> = {
  docx: "DOC",
  pdf: "PDF",
  txt: "TXT",
  md: "MD",
};

interface Props {
  items: AttachmentMeta[];
  onRemove: (id: string) => void;
  onReorder: (ids: string[]) => void;
  disabled?: boolean;
}

export function AttachmentCards({ items, onRemove, onReorder, disabled }: Props) {
  const [dragging, setDragging] = useState<string | null>(null);
  const [over, setOver] = useState<string | null>(null);

  if (items.length === 0) return null;

  const move = (from: string, to: string) => {
    if (from === to) return;
    const ids = items.map((i) => i.id);
    const a = ids.indexOf(from);
    const b = ids.indexOf(to);
    if (a < 0 || b < 0) return;
    ids.splice(b, 0, ids.splice(a, 1)[0]);
    onReorder(ids);
  };

  return (
    <ul
      className="att-list"
      aria-label="已导入的文档"
      onDragOver={(e) => {
        // the list itself accepts drops so dragging past the last card still
        // lands somewhere meaningful
        if (dragging) e.preventDefault();
      }}
      onDrop={(e) => {
        if (!dragging) return;
        e.preventDefault();
        setDragging(null);
        setOver(null);
      }}
    >
      {items.map((item, index) => (
        <li
          key={item.id}
          className={
            "att-card" +
            (dragging === item.id ? " dragging" : "") +
            (over === item.id && dragging && dragging !== item.id ? " over" : "")
          }
          draggable={!disabled}
          onDragStart={(e) => {
            if (disabled) return;
            e.dataTransfer.effectAllowed = "move";
            e.dataTransfer.setData("text/plain", item.id);
            setDragging(item.id);
          }}
          onDragEnter={() => {
            if (dragging) setOver(item.id);
          }}
          onDragOver={(e) => {
            if (dragging) e.preventDefault();
          }}
          onDrop={(e) => {
            if (!dragging) return;
            e.preventDefault();
            e.stopPropagation();
            move(dragging, item.id);
            setDragging(null);
            setOver(null);
          }}
          onDragEnd={() => {
            setDragging(null);
            setOver(null);
          }}
          title={item.notes.length ? `${item.name}\n${item.notes.join("\n")}` : item.name}
        >
          <span className={"att-badge att-" + item.ext}>{BADGE[item.ext] ?? item.ext.toUpperCase()}</span>
          <span className="att-name">{item.name}</span>
          <span className="att-chars">
            {item.chars.toLocaleString("zh-CN")} 字
            {item.notes.length > 0 && <span className="att-note">!</span>}
          </span>
          <span className="att-order" aria-hidden="true">
            {index + 1}
          </span>
          <button
            type="button"
            className="att-x"
            aria-label={`移除 ${item.name}`}
            disabled={disabled}
            onClick={() => onRemove(item.id)}
          >
            <svg viewBox="0 0 12 12" width="10" height="10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
              <path d="M2.5 2.5 9.5 9.5M9.5 2.5 2.5 9.5" />
            </svg>
          </button>
        </li>
      ))}
    </ul>
  );
}
