// Per-sentence detail list (2026-11-21 rework): one flat, filtered, sorted
// list inside the results panel instead of one block per engine. The model
// capsule above the document decides whose scores these are; "综合" shows the
// per-sentence max with every engine's number behind the rate badge.
//
// round-12: hover tooltips are gone (unreliable to hit, invisible on touch) -
// the percentage is now a button that opens a spring popover listing which
// model said what. The whole list collapses with the same spring as the
// composite detail card.
import { useMemo, useState } from "react";
import { bucketColor, type HighlightMode, type SentenceSpan } from "./highlight";
import { Collapse } from "./Collapse";
import { Popover } from "./Popover";
import { createCodePointView } from "./textOffsets";

export interface SentenceRow {
  span: SentenceSpan;
  /** 1-based position in the original document, shown even when rate-sorted */
  index: number;
}

interface Props {
  text: string;
  rows: SentenceRow[];
  total: number;
  mode: HighlightMode;
  open: boolean;
  onToggle: () => void;
  /** active filter/sort sentence, e.g. 只看 AI 率较高 · 按 AI 率降序 */
  note: string;
  /** engine id -> friendly catalogue name */
  nameOf: (id: string) => string;
}

function RateBadge({
  span,
  mode,
  nameOf,
}: {
  span: SentenceSpan;
  mode: HighlightMode;
  nameOf: (id: string) => string;
}) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  const [open, setOpen] = useState(false);
  const color = bucketColor(span.score, mode);
  const sources = [...span.sources].sort((a, b) => b.score - a.score);

  return (
    <>
      <button
        type="button"
        className="sent-rate"
        style={color ? { background: color } : undefined}
        aria-expanded={open}
        aria-label={`这一句 ${Math.round(span.score * 100)}%，点开看各模型分数`}
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
        {Math.round(span.score * 100)}%
      </button>
      <Popover
        open={open}
        anchor={anchor}
        width={252}
        onClose={() => setOpen(false)}
        ariaLabel="这一句的模型分数"
      >
        <div className="pop-head">这一句 · {sources.length} 个模型</div>
        {sources.map((s) => (
          <div className="pop-row" key={s.engine}>
            <span className="pop-row-name">{nameOf(s.engine)}</span>
            <span className="pop-pct" style={{ background: bucketColor(s.score, mode) ?? undefined }}>
              {Math.round(s.score * 100)}%
            </span>
          </div>
        ))}
        <p className="pop-foot">分数 = 该引擎对这句话给出的 AI 概率；综合显示其中最高值。</p>
      </Popover>
    </>
  );
}

export function SentenceList({ text, rows, total, mode, open, onToggle, note, nameOf }: Props) {
  const view = useMemo(() => createCodePointView(text), [text]);
  const pageSize = 100;
  const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
  const [pageState, setPageState] = useState({ rows, open, page: 0 });
  const currentPage =
    pageState.rows === rows && pageState.open === open
      ? Math.min(pageState.page, pageCount - 1)
      : 0;

  const pageRows = rows.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

  const renderRows = () => (
    <>
      <ul className="sent-list">
        {pageRows.map(({ span, index }) => {
          const body = view.slice(span.offset, span.offset + span.length).trim();
          return (
            <li key={`${span.offset}:${span.length}`}>
              <span className="sent-no">{index}</span>
              <span className="sent-text">“{body}”</span>
              <RateBadge span={span} mode={mode} nameOf={nameOf} />
            </li>
          );
        })}
        {rows.length === 0 && (
          <li className="sent-empty">
            {total === 0
              ? "没有可打分的句子（文本过短或引擎未返回句级结果）。"
              : "该筛选条件下没有句子——试试「全部句子」。"}
          </li>
        )}
      </ul>
      {pageCount > 1 && (
        <div className="sent-pager" aria-label="逐句明细分页">
          <button
            type="button"
            className="sent-page-button"
            disabled={currentPage === 0}
            onClick={() =>
              setPageState({ rows, open, page: Math.max(0, currentPage - 1) })
            }
          >
            上一页
          </button>
          <span>
            {currentPage + 1} / {pageCount} 页 · 每页 100 句
          </span>
          <button
            type="button"
            className="sent-page-button"
            disabled={currentPage >= pageCount - 1}
            onClick={() =>
              setPageState({
                rows,
                open,
                page: Math.min(pageCount - 1, currentPage + 1),
              })
            }
          >
            下一页
          </button>
        </div>
      )}
    </>
  );

  return (
    <section className="sent-block">
      <div className="sent-head">
        <span className="sent-title">逐句明细</span>
        <span className="hint">
          {rows.length}/{total} 句 · {note}
        </span>
        <button type="button" className="sent-toggle" onClick={onToggle} aria-expanded={open}>
          {open ? "收起" : "展开"}
        </button>
      </div>
      <Collapse open={open} unmountOnClose render={renderRows} />
    </section>
  );
}
