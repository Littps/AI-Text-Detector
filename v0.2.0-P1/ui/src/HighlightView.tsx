// D28 highlight view: whole-text render with per-span background colours.
// Round-12: the hover tooltip is gone - per-engine numbers live behind the
// rate badge in the sentence list, so the text stays selectable and quiet.
// Round-30: the frame got a top-right reading toggle. The button is sticky
// inside the scroller, so it stays reachable on a long document without
// stealing a line of its own (zero-height tool row).
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { bucketColor, legendFor, type HighlightMode, type SentenceSpan } from "./highlight";
import { createCodePointView } from "./textOffsets";

interface Props {
  text: string;
  spans: SentenceSpan[];
  mode: HighlightMode;
}

/**
 * Render the document as (plain | colored) segments. Offsets are code
 * points; the code-point view keeps emoji/astral chars aligned without
 * materialising the whole document as an array.
 */
export function HighlightView({ text, spans, mode }: Props) {
  // reading mode: bigger type + looser leading, not a different layout - the
  // user asked for "较大的行高来看", so the line-height carries most of it
  const [zoom, setZoom] = useState(false);
  const textKey = `${text.length}:${text.charCodeAt(0) || 0}:${text.charCodeAt(text.length - 1) || 0}`;
  const [pageState, setPageState] = useState({ textKey, page: 0 });
  const frameRef = useRef<HTMLDivElement | null>(null);
  const view = useMemo(() => createCodePointView(text), [text]);
  const pageSize = 40_000;
  const pageCount = Math.max(1, Math.ceil(view.length / pageSize));
  const currentPage =
    pageState.textKey === textKey ? Math.min(pageState.page, pageCount - 1) : 0;
  const pageStart = currentPage * pageSize;
  const pageEnd = Math.min(view.length, pageStart + pageSize);

  useEffect(() => {
    frameRef.current?.scrollTo({ top: 0 });
  }, [currentPage]);

  const parts: { color: string | null; start: number; end: number; span?: SentenceSpan }[] = [];
  let cursor = pageStart;
  for (const s of spans) {
    const spanEnd = s.offset + s.length;
    if (spanEnd <= pageStart) continue;
    if (s.offset >= pageEnd) break;
    const start = Math.max(pageStart, s.offset);
    const end = Math.min(pageEnd, spanEnd);
    if (end <= start) continue;
    if (start > cursor) parts.push({ color: null, start: cursor, end: start });
    parts.push({ color: bucketColor(s.score, mode), start, end, span: s });
    cursor = end;
  }
  if (cursor < pageEnd) parts.push({ color: null, start: cursor, end: pageEnd });

  return (
    <div className="highlight-area">
      <div ref={frameRef} className={"highlight-frame" + (zoom ? " zoom" : "")}>
        <p className="highlight-text">
          {/* round-30: floated into the paragraph's top-right corner, so the
              first line wraps around it - a zero-height sticky overlay looked
              tidier but covered the end of the first line (r30 screenshot). */}
          <button
            type="button"
            className={"hl-zoom" + (zoom ? " on" : "")}
            onClick={() => setZoom((v) => !v)}
            aria-pressed={zoom}
            aria-label={zoom ? "还原字号与行高" : "放大字号与行高"}
            title={zoom ? "还原字号与行高" : "放大字号与行高，便于逐句细读"}
          >
            <svg
              viewBox="0 0 16 16"
              width="12"
              height="12"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="hl-zoom-icon"
            >
              <path d="M9.6 2.6h3.8v3.8" />
              <path d="M13.4 2.6 8.8 7.2" />
              <path d="M6.4 13.4H2.6V9.6" />
              <path d="M2.6 13.4 7.2 8.8" />
            </svg>
          </button>
          {parts.map((p, i) => {
            const slice = view.slice(p.start, p.end);
            if (p.color === null) return <span key={i}>{slice}</span>;
            // the band colour travels as --hl-c: an inline `background` shorthand
            // would reset the gradient layer .hl uses to keep the band slim.
            return (
              <span
                key={`${p.start}:${p.end}`}
                className="hl"
                style={{ "--hl-c": p.color } as CSSProperties}
              >
                {slice}
              </span>
            );
          })}
        </p>
      </div>
      {pageCount > 1 && (
        <div className="highlight-pager" aria-label="正文分段">
          <button
            type="button"
            className="highlight-page-button"
            disabled={currentPage === 0}
            onClick={() =>
              setPageState({ textKey, page: Math.max(0, currentPage - 1) })
            }
            aria-label="上一段正文"
          >
            ‹
          </button>
          <span>
            第 {currentPage + 1} / {pageCount} 段 · 每段约 4 万字
          </span>
          <button
            type="button"
            className="highlight-page-button"
            disabled={currentPage >= pageCount - 1}
            onClick={() =>
              setPageState({
                textKey,
                page: Math.min(pageCount - 1, currentPage + 1),
              })
            }
            aria-label="下一段正文"
          >
            ›
          </button>
        </div>
      )}
      <p className="legend legend-buckets">
        {legendFor(mode).map((b) => (
          <span key={b.label} className="legend-item">
            <span className="legend-swatch" style={{ background: b.color }} />
            {b.label}
          </span>
        ))}
      </p>
    </div>
  );
}
