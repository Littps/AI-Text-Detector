// Composite result card (2026-11-21 user design): the whole-document AI rate
// now leads the results panel instead of sitting as the last table row. The
// triangle at its bottom-left unfolds the calculation detail: per-engine doc
// rates plus the exact rule used for the number.
import { bucketColor, bucketLabel, type HighlightMode } from "./highlight";
import { Collapse } from "./Collapse";
import { SummaryTable } from "./SummaryTable";
import type { DetectResult } from "./types";

interface Props {
  score: number | null;
  mode: HighlightMode;
  /** ok engines only */
  results: DetectResult[];
  spanCount: number;
  detailOpen: boolean;
  onToggleDetail: () => void;
  nameOf: (id: string) => string;
}

export function CompositeCard({
  score,
  mode,
  results,
  spanCount,
  detailOpen,
  onToggleDetail,
  nameOf,
}: Props) {
  const color = score === null ? null : bucketColor(score, mode);
  return (
    <section className={"composite-card" + (detailOpen ? " open" : "")}>
      <div className="composite-main">
        <span className="composite-score">{score === null ? "—" : `${Math.round(score * 100)}%`}</span>
        <div className="composite-meta">
          <span className="composite-title">综合 AI 率</span>
          <span className="composite-tag">
            {color && <span className="composite-dot" style={{ background: color }} />}
            {score === null ? "无可用句级结果" : bucketLabel(score, mode)}
          </span>
          <span className="composite-sub">
            {results.length} 个引擎 · 覆盖 {spanCount} 句 · 逐句取各引擎最高分后按句长加权
          </span>
        </div>
      </div>

      <Collapse open={detailOpen}>
        <div className="composite-detail">
          <SummaryTable results={results} mode={mode} nameOf={nameOf} />
          <p className="hint">
            口径：每个句子取所有引擎里最高的 AI 率（宁可误报不漏报），再按句长加权平均；
            文档级引擎（整篇一个分数）不参与句级加权，只出现在上表。
          </p>
        </div>
      </Collapse>

      <button type="button" className="composite-toggle" onClick={onToggleDetail} aria-expanded={detailOpen}>
        {/* drawn as geometry, not as the ▸ glyph: at 10px the text character
            fell back to a symbol font whose top half never got rasterised.
            The path is deliberately wider than the old 3.8/5.4 sliver so it
            still reads as a triangle at 100% Windows scaling. */}
        <svg
          className={"tri" + (detailOpen ? " open" : "")}
          width="10"
          height="10"
          viewBox="0 0 10 10"
          aria-hidden="true"
        >
          <path
            d="M2.9 1.9 L8.3 5 L2.9 8.1 Z"
            fill="currentColor"
            stroke="currentColor"
            strokeWidth="1.1"
            strokeLinejoin="round"
          />
        </svg>
        {detailOpen ? "收起计算详情" : "计算详情"}
      </button>
    </section>
  );
}
