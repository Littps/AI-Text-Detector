// D28/D29 highlight domain: four-state bucketing, max sentence
// aggregation, whole-doc composite. Pure functions - no React here.
//
// Scores are AI-probabilities (0..1). Backend offsets are Python code
// points; textOffsets.ts maps them without materialising every character.
import type { DetectResult, DetectSegment, StorageMode } from "./types";
import { codePointLength } from "./textOffsets";

export type HighlightMode = "gradient6" | "three" | "two" | "off";

export const HIGHLIGHT_MODES: { id: HighlightMode; label: string }[] = [
  { id: "gradient6", label: "六度渐变（默认）" },
  { id: "three", label: "简单三度" },
  { id: "two", label: "极简两度" },
  { id: "off", label: "关闭高亮" },
];

export interface Bucket {
  min: number;
  color: string;
  label: string;
}

// D28 color values are pinned: 红/粉红/橙/浅橙/浅黄/无.
const GRADIENT6: Bucket[] = [
  { min: 0.9, color: "#F09595", label: "几乎确定 AI（≥90%）" },
  { min: 0.75, color: "#F7C1C1", label: "很可能 AI（75–90%）" },
  { min: 0.6, color: "#FBCBA8", label: "偏向 AI（60–75%）" },
  { min: 0.45, color: "#FAE3C8", label: "不确定（45–60%）" },
  { min: 0.25, color: "#FAEEDA", label: "轻微 AI 倾向（25–45%）" },
];

const THREE: Bucket[] = [
  { min: 0.75, color: "#F09595", label: "很可能 AI（≥75%）" },
  { min: 0.5, color: "#FBCBA8", label: "疑似 AI（50–75%）" },
];

const TWO: Bucket[] = [{ min: 0.5, color: "#F09595", label: "AI 倾向（≥50%）" }];

const BUCKETS: Record<HighlightMode, Bucket[] | null> = {
  gradient6: GRADIENT6,
  three: THREE,
  two: TWO,
  off: null,
};

/** Background color for a score, or null when the span stays uncolored. */
export function bucketColor(score: number, mode: HighlightMode): string | null {
  const buckets = BUCKETS[mode];
  if (!buckets) return null;
  for (const b of buckets) {
    if (score >= b.min) return b.color;
  }
  return null;
}

/** Human-readable bucket for the composite card ("很可能 AI（75–90%）"). */
export function bucketLabel(score: number, mode: HighlightMode): string {
  const buckets = BUCKETS[mode];
  if (!buckets) return "高亮已关闭";
  for (const b of buckets) {
    if (score >= b.min) return b.label;
  }
  return mode === "gradient6" ? "人类（<25%）" : "人类（<50%）";
}

export function legendFor(mode: HighlightMode): { color: string; label: string }[] {
  const buckets = BUCKETS[mode];
  if (!buckets) return [];
  const out = buckets.map((b) => ({ color: b.color, label: b.label }));
  if (mode === "gradient6") out.push({ color: "transparent", label: "人类（<25%）" });
  if (mode === "three") out.push({ color: "transparent", label: "人类（<50%）" });
  if (mode === "two") out.push({ color: "transparent", label: "人类（<50%）" });
  return out;
}

// ---- localStorage persistence (P4 may move it to the sidecar) ----------

const LS_KEY = "aigc.highlight.mode";

export function loadHighlightMode(): HighlightMode {
  try {
    const v = localStorage.getItem(LS_KEY);
    if (v && v in BUCKETS) return v as HighlightMode;
  } catch {
    /* private mode etc. - default stays */
  }
  return "gradient6";
}

export function saveHighlightMode(mode: HighlightMode): void {
  try {
    localStorage.setItem(LS_KEY, mode);
  } catch {
    /* non-fatal */
  }
}

// ---- sentence-level max aggregation (D29: 宁误报不漏报) ------------------

export interface SpanSource {
  engine: string;
  score: number;
}

export interface SentenceSpan {
  offset: number;
  length: number;
  score: number; // max across engines on this span
  sources: SpanSource[];
}

/**
 * Merge every engine's segments into per-span max scores (same span from
 * several engines -> keep the highest, keep all sources for the tooltip).
 * Document-level segments (span == whole text) are excluded here: they
 * belong to the summary table only, not sentence coloring.
 */
export function aggregateMax(
  results: DetectResult[],
  docLength: number,
): SentenceSpan[] {
  const byKey = new Map<string, SentenceSpan>();
  for (const r of results) {
    if (r.status !== "ok") continue;
    for (const s of r.segments as DetectSegment[]) {
      if (s.length >= docLength) continue; // doc-level segment
      const key = `${s.offset}:${s.length}`;
      const cur = byKey.get(key);
      if (cur) {
        cur.sources.push({ engine: r.engine, score: s.score });
        if (s.score > cur.score) cur.score = s.score;
      } else {
        byKey.set(key, {
          offset: s.offset,
          length: s.length,
          score: s.score,
          sources: [{ engine: r.engine, score: s.score }],
        });
      }
    }
  }
  return [...byKey.values()].sort((a, b) => a.offset - b.offset);
}

/** Whole-doc composite over the max-aggregated view: length-weighted mean. */
export function overallScore(spans: SentenceSpan[]): number | null {
  if (spans.length === 0) return null;
  const totalLen = spans.reduce((acc, s) => acc + s.length, 0);
  if (totalLen <= 0) return null;
  return (
    Math.round((spans.reduce((acc, s) => acc + s.score * s.length, 0) / totalLen) * 1000) / 1000
  );
}

/** Per-engine doc rates for the summary table (backend-computed). */
export function engineDocScores(results: DetectResult[]): { engine: string; score: number | null; status: string }[] {
  return results.map((r) => ({ engine: r.engine, score: r.doc_score, status: r.status }));
}

// ---- language warning chain (D22 copy) -----------------------------------

export function languageWarning(text: string, results: DetectResult[]): string | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  let zhChars = 0;
  for (const char of trimmed) {
    if (/[\u4e00-\u9fff]/.test(char)) zhChars += 1;
  }
  const mostlyZh = zhChars > codePointLength(trimmed) * 0.3;
  if (!mostlyZh) return null;
  const enEngines = results
    .filter((r) => r.status === "ok" && r.engine.endsWith("-en"))
    .map((r) => r.engine);
  if (enEngines.length === 0) return null;
  return `输入以中文为主，但 ${enEngines.join("、")} 是英文模型——它对中文的分数没有参考价值，请忽略其结果。`;
}

// ---- storage helpers (D25 UI copy) ---------------------------------------

export const STORAGE_MODE_LABEL: Record<StorageMode, string> = {
  "keep-original": "保留原件（默认）",
  dual: "双份（fp32+fp16）",
  "fp16-only": "仅半精度（省空间）",
};

export function storageBadge(s: {
  fp32_present: boolean;
  fp16_present: boolean;
}): string {
  if (s.fp32_present && s.fp16_present) return "fp32+fp16";
  if (s.fp32_present) return "fp32";
  if (s.fp16_present) return "fp16";
  return "未下载";
}
