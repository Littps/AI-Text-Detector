// D29 summary table: per-engine whole-doc AI rate. The composite row moved
// out to CompositeCard (2026-11-21) - it is no longer the last row here.
import { bucketColor, type HighlightMode } from "./highlight";
import type { DetectResult } from "./types";

interface Props {
  results: DetectResult[];
  mode: HighlightMode;
  nameOf: (id: string) => string;
}

const STATUS_LABEL: Record<string, string> = {
  ok: "正常",
  weights_missing: "权重未下载",
  error: "出错",
  unknown_engine: "未知引擎",
};

export function SummaryTable({ results, mode, nameOf }: Props) {
  return (
    <table className="summary-table">
      <thead>
        <tr>
          <th>引擎</th>
          <th>整篇 AI 率</th>
          <th>状态</th>
        </tr>
      </thead>
      <tbody>
        {results.map((r) => {
          const pct = r.doc_score === null ? null : Math.round(r.doc_score * 100);
          const color = r.doc_score === null ? null : bucketColor(r.doc_score, mode);
          return (
            <tr key={r.engine}>
              <td>{nameOf(r.engine)}</td>
              <td>
                <span className="doc-rate" style={color ? { background: color } : undefined}>
                  {pct === null ? "—" : `${pct}%`}
                </span>
              </td>
              <td>
                {STATUS_LABEL[r.status] ?? r.status}
                {r.detail ? `（${r.detail}）` : ""}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
