// P3-a functional UI (D28/D29/D30) + P3-b Claude skin (D28④/D28⑤/D30①)
// + P3-b round-2 UX rework (2026-11-18 user feedback):
//   - paste/detect is the protagonist: it leads the page
//   - a real settings section (highlight mode + glass preference)
//   - model management and connection fold away (Tauri auto-connects)
//   - per-sentence details gain forward/reverse filtering (all / AI-ish /
//     human-ish), default shows every sentence with its probability
//   - router engines are labeled on their chip (no AI score by design)
// Glass degradation stays a pure CSS-state swap via [data-glass].
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ClipboardEvent as ReactClipboardEvent,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  RETENTION_OPTIONS,
  RISK_DUAL_DIALOG,
  type AttachmentMeta,
  type DetectJobInfo,
  type DetectPayload,
  type DetectResult,
  type DetectSegment,
  type DlEvent,
  type DlStatusPayload,
  type DownloadSource,
  type EnginesPayload,
  type FileImportPayload,
  type HealthPayload,
  type RetentionTier,
  type ResidentEntry,
  type RuntimeStatusPayload,
  type UnloadPayload,
} from "./types";
import { AttachmentCards } from "./AttachmentCards";
import {
  inTauri,
  MAX_BYTES,
  oversized,
  pickBrowserFiles,
  pickPaths,
  toBlobs,
  type FileBlob,
} from "./filepicker";
import { RiskDialog } from "./RiskDialog";
import { StartupGate } from "./StartupGate";
import { HealthLight, deriveHealth } from "./HealthLight";
import { DevicePanel } from "./DevicePanel";
import { HighlightView } from "./HighlightView";
import { StoragePanel } from "./StoragePanel";
import { RuntimePanel } from "./RuntimePanel";
import { TitleBar } from "./TitleBar";
import {
  HIGHLIGHT_MODES,
  aggregateMax,
  languageWarning,
  loadHighlightMode,
  overallScore,
  saveHighlightMode,
  type HighlightMode,
  type SentenceSpan,
} from "./highlight";
import {
  GLASS_MODES,
  loadGlassPref,
  resolveGlass,
  saveGlassPref,
  type GlassPreference,
} from "./glass";
import { SidecarClient, fmtBytes, fmtMb, type ProgressFrame } from "./ws";
import { ModelModePopover } from "./ModelModePopover";
import { DevOptions, loadDevUnlocked } from "./DevGate";
import { LogExport } from "./LogExport";
import { Notices, type NoticeItem } from "./Notice";
import { CompositeCard } from "./CompositeCard";
import { InfoCapsule } from "./InfoCapsule";
import { SentenceList, type SentenceRow } from "./SentenceList";
import { SelectCapsule } from "./SelectCapsule";
import { ContextMenu, type MenuItem } from "./ContextMenu";
import { codePointLength } from "./textOffsets";
import {
  copySelection,
  cutSelection,
  hasSelectionIn,
  pasteInto,
  selectAllIn,
  textFieldOf,
  type TextTarget,
} from "./clipboard";
import type {
  CustomImportResult,
  CustomJob,
  CustomListPayload,
} from "./types";
import {
  DETECT_LANGS,
  detectLangLabel,
  detectLangOrNull,
  ENGINE_LANG_TAG,
  langWithNoun,
  MODE_LABELS,
  resolveModeEngines,
  usableEngineIds,
  type DetectLang,
  type DetectMode,
} from "./modelModes";

interface ProgressEntry {
  done: number;
  total: number;
  dest: string;
}

// Per-sentence filter thresholds align with the D28 buckets: >=75% is the
// "very likely AI" bucket, <25% is the human bucket of the six-step gradient.
type SentenceFilter = "all" | "ai" | "human";

/** Why a paste left the composer empty. Shown as a modal on the detect click
 *  (round-14 user rule: pasting never asks for permission, but the app must say
 *  out loud when the clipboard held something it cannot read). */
type PasteIssue = "image" | "file" | "nontext";
const SENTENCE_FILTERS: { id: SentenceFilter; label: string; hint: string }[] = [
  { id: "all", label: "全部句子（默认）", hint: "不筛句，按下面的排序偏好" },
  { id: "ai", label: "只看 AI 率较高", hint: "≥75%，从高到低" },
  { id: "human", label: "只看比较像人工", hint: "<25%，最像人工的在前" },
];

// Sentence order preference (settings -> 常规): the document's own order by
// default; "rate" flips the 全部句子 list to AI-rate-descending.
type SortPref = "text" | "rate";
const SORT_KEY = "aigc.sentence.sort";

function loadSortPref(): SortPref {
  try {
    return localStorage.getItem(SORT_KEY) === "rate" ? "rate" : "text";
  } catch {
    return "text";
  }
}

// Download source preference (D36, settings -> 常规): "official" keeps the
// D18 default (huggingface.co first), "mirror" leads with hf-mirror.com for
// networks that cannot reach the official host without a proxy. Either way
// the other host stays in the chain, so a preference never costs a retry.
const SOURCE_KEY = "aigc.download.source";
/** Mirrors the backend attachment/compose ceiling; browser text is measured
 * in UTF-16 units here, which is conservative for astral characters. */
const MAX_INPUT_CHARS = 4_000_000;

const SOURCE_OPTIONS: { id: DownloadSource; label: string; hint: string; host: string }[] = [
  {
    id: "official",
    label: "优先使用官方源（默认）",
    hint: "模型先走 Hugging Face，运行时先走 PyTorch / PyPI 官方；失败后切换可用镜像。",
    host: "huggingface.co",
  },
  {
    id: "mirror",
    label: "优先使用镜像源",
    hint: "模型先走 hf-mirror.com，Python 包优先国内 PyPI 镜像；cu129 无镜像时仍走官方源。",
    host: "hf-mirror.com",
  },
];

function loadDownloadSource(): DownloadSource {
  try {
    const v = localStorage.getItem(SOURCE_KEY);
    if (SOURCE_OPTIONS.some((o) => o.id === v)) return v as DownloadSource;
  } catch {
    /* default below */
  }
  return "official";
}

function sourceHost(id: DownloadSource | undefined): string {
  return SOURCE_OPTIONS.find((o) => o.id === id)?.host ?? SOURCE_OPTIONS[0].host;
}

// Page transition (round-13): switching 工作台 / 模型管理 / 设置 used to swap the
// whole column in one frame. The nav highlight still flips on the click frame
// (input has to feel instant), while the content pane plays a short exit and
// then springs in from the direction of travel — so the two sections read as
// one surface sliding rather than two pages blinking.
type ViewId = "work" | "models" | "runtime" | "settings";
/** must stay in sync with the `view-out` keyframe duration in index.css */
const VIEW_EXIT_MS = 140;

// Residency tier persistence: the sidecar re-applies it on every connect, so
// a restart keeps the user's choice (the backend itself stays stateless).
const RETENTION_KEY = "aigc.resident.retention";

function loadRetention(): RetentionTier {
  try {
    const v = localStorage.getItem(RETENTION_KEY);
    if (RETENTION_OPTIONS.some((o) => o.id === v)) return v as RetentionTier;
  } catch {
    /* default below */
  }
  return "default";
}

// Engine pick persistence (round-2 bugfix: an empty pick set silently
// disabled the detect button, which read as "long text can't run").
const ENGINE_PICK_KEY = "aigc.detect.engines";

function loadSavedEnginePick(): string[] {
  try {
    const raw = localStorage.getItem(ENGINE_PICK_KEY);
    const arr: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

function saveEnginePick(ids: Iterable<string>): void {
  try {
    localStorage.setItem(ENGINE_PICK_KEY, JSON.stringify([...ids]));
  } catch {
    /* private mode / quota — picks just won't persist */
  }
}

// Detect-mode persistence (round-6): last chosen mode + precise-language pick.
const MODE_KEY = "aigc.detect.mode";
const PRECISE_LANG_KEY = "aigc.detect.preciseLang";

function loadMode(): DetectMode {
  try {
    const v = localStorage.getItem(MODE_KEY);
    return v === "fast" || v === "balanced" || v === "precise" || v === "custom" ? v : "balanced";
  } catch {
    return "balanced";
  }
}

function loadPreciseLang(): "zh" | "en" {
  try {
    return localStorage.getItem(PRECISE_LANG_KEY) === "en" ? "en" : "zh";
  } catch {
    return "zh";
  }
}

function passesFilter(score: number, f: SentenceFilter): boolean {
  if (f === "ai") return score >= 0.75;
  if (f === "human") return score < 0.25;
  return true;
}

// Mirror-hint hosts for the custom URL download prompt (aligned with the
// backend custom_repo._MIRRORS single source of truth).
const MIRROR_HOSTS = ["huggingface.co"];

/**
 * Custom-library timestamps arrive as ISO-8601 in UTC (`+00:00`). Render them
 * in the local zone: slicing the raw string printed 16:29 for an import that
 * happened at 00:29 the next day on UTC+8 (calendar date was wrong too).
 */
function fmtImportedAt(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.replace("T", " ").slice(0, 16);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

/** Time-based greeting in Claude's tone. */
function greetingText(): string {
  const h = new Date().getHours();
  if (h < 5) return "深夜即兴时刻。";
  if (h < 11) return "早晨好。";
  if (h < 14) return "中午好。";
  if (h < 18) return "下午好。";
  if (h < 23) return "晚上好。";
  return "深夜即兴时刻。";
}

export default function App() {
  // connection
  const [port, setPort] = useState("21000");
  const [token, setToken] = useState("dev-token");
  const [state, setState] = useState<"connecting" | "open" | "closed" | "idle">("idle");
  const clientRef = useRef<SidecarClient | null>(null);

  // data
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [catalog, setCatalog] = useState<EnginesPayload | null>(null);
  const [dl, setDl] = useState<DlStatusPayload | null>(null);
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatusPayload | null>(null);
  const [progress, setProgress] = useState<Record<string, ProgressEntry>>({});
  const [castProgress, setCastProgress] = useState<Record<string, ProgressFrame>>({});
  const [jobLog, setJobLog] = useState<string[]>([]);

  // selection + D20 gate
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [gate, setGate] = useState<{ engineId: string; riskLevel: string; riskTextKey: string | null } | null>(null);

  // detect (D29 streaming)
  const [text, setText] = useState("");
  const [textAtDetect, setTextAtDetect] = useState(""); // stale detection anchor
  // D40: with attachments the scored text is text + file bodies, so comparing
  // the rendered text would report "changed" forever. Keep the original text
  // and ordered ids separately; this avoids building a second multi-megabyte
  // concatenated signature on every render.
  const [inputAtDetect, setInputAtDetect] = useState({
    text: "",
    attachments: "",
  });
  const [detectEngines, setDetectEngines] = useState<Set<string>>(new Set());
  const [detectResult, setDetectResult] = useState<DetectPayload | null>(null);
  const [detecting, setDetecting] = useState(false);
  // the orphan poller below reads this from a timer closure: a run started in
  // this page must never be reported as somebody else's orphan
  const detectingRef = useRef(false);
  useEffect(() => {
    detectingRef.current = detecting;
  }, [detecting]);
  // round-19: a run that outlived its window (reload mid-detect). The strip
  // below the composer names it and offers the only two honest options.
  const [jobNotice, setJobNotice] = useState<DetectJobInfo | null>(null);
  const [bootSeconds, setBootSeconds] = useState(0);
  const [partials, setPartials] = useState<Record<string, DetectSegment[]>>({});
  const partialsRef = useRef<Record<string, DetectSegment[]>>({});
  const partialFlush = useRef<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // highlight mode (D28 four states, localStorage-persisted)
  const [hlMode, setHlMode] = useState<HighlightMode>(() => loadHighlightMode());

  // glass preference (P3-b D28⑤: auto follows tier+busy, on/off override)
  const [glassPref, setGlassPref] = useState<GlassPreference>(() => loadGlassPref());

  // per-sentence filter (round-2 feedback: forward/reverse screening)
  const [sentFilter, setSentFilter] = useState<SentenceFilter>("all");
  // 2026-11-21 results rework: whose scores are shown, how they are ordered,
  // and whether the sentence list / composite detail are unfolded
  const [modelFilter, setModelFilter] = useState<string>("all");
  const [sortPref, setSortPref] = useState<SortPref>(() => loadSortPref());
  const [compDetail, setCompDetail] = useState(false);
  const [sentOpen, setSentOpen] = useState(false);
  const [composerTall, setComposerTall] = useState(false);
  const [retention, setRetention] = useState<RetentionTier>(() => loadRetention());
  const retentionRef = useRef<RetentionTier>(retention);
  // D36 engine-download source preference (settings -> 常规); sent with every
  // download/start so the backend never has to remember client state.
  const [downloadSource, setDownloadSource] = useState<DownloadSource>(() =>
    loadDownloadSource(),
  );
  // custom right-click menu + transient toast
  // seq forces a fresh ContextMenu instance per open: reopening at the same
  // point while the previous exit animation still runs would otherwise reuse
  // the closing instance and stay invisible (found in round-10 automation)
  const [ctxMenu, setCtxMenu] = useState<{
    seq: number;
    x: number;
    y: number;
    target: TextTarget | null;
    items: MenuItem[];
  } | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  // round-30: bottom-right notices (model unloaded, log exported). Separate
  // from `toast` on purpose - they carry a result the user asked for, live
  // 15 s and can be dismissed individually.
  const [notices, setNotices] = useState<NoticeItem[]>([]);
  const noticeSeq = useRef(0);
  // which resident chip is mid-unload (spinner replaces the ×)
  const [unloading, setUnloading] = useState<string | null>(null);
  // round-32: which language's engine set the capsule popover lists. The
  // language actually used for detection is still decided by the text (or by
  // precise's confirmation) — this is a preview switch, nothing more.
  const [previewLang, setPreviewLang] = useState<DetectLang>("zh");
  // round-14: paste diagnostics (never a permission prompt, but never silent
  // either when the clipboard held an image/file instead of text)
  const [pasteIssue, setPasteIssue] = useState<PasteIssue | null>(null);
  const [formatOpen, setFormatOpen] = useState(false);

  // D40 file import: the sidecar keeps the parsed text, the UI keeps only the
  // metadata (id/name/chars) in list order — that order is the compose order.
  const [attachments, setAttachments] = useState<AttachmentMeta[]>([]);
  // round-20: the orphan-adopt path rebuilds the input signature from this
  // list, but it runs from a timer effect that captured an older render —
  // read it through a ref so a card added since then is not missed.
  const attachmentsRef = useRef<AttachmentMeta[]>([]);
  useEffect(() => {
    attachmentsRef.current = attachments;
  }, [attachments]);
  const [attBusy, setAttBusy] = useState(false);
  const [attErrors, setAttErrors] = useState<{ name: string; error: string }[]>([]);
  const [dropHot, setDropHot] = useState(false);
  // first import while the box has text asks once whether to keep it (the
  // composed document is text-first, then attachments in card order)
  const [importAsk, setImportAsk] = useState<{ chars: number; run: (keepText: boolean) => void } | null>(null);
  const composerRef = useRef<HTMLDivElement | null>(null);

  // round-6: workbench / model-management / settings separation
  // round-13: `view` drives the nav highlight and data effects (immediate);
  // `shown` is what is actually painted, lagging one exit animation behind.
  const [view, setView] = useState<ViewId>("work");
  const [shown, setShown] = useState<ViewId>("work");
  const [paneLeaving, setPaneLeaving] = useState(false);
  const shownRef = useRef<ViewId>("work");
  const exitTimer = useRef<number | null>(null);
  const [mode, setMode] = useState<DetectMode>(() => loadMode());
  const [preciseLang, setPreciseLang] = useState<"zh" | "en">(() => loadPreciseLang());
  const [modeOpen, setModeOpen] = useState(false);
  // popover anchor: a stable ref to the capsule button — the popover reads
  // its rect itself on mount/sub-switch (more robust than passing a rect
  // through click-handler state)
  const capsuleRef = useRef<HTMLButtonElement | null>(null);

  // Nav click -> exit the current pane, then swap and let the new one spring
  // in. Re-clicking the active item cancels a pending swap instead of stacking
  // timers; rapid A->B->C clicks keep only the last destination.
  const switchView = useCallback((next: ViewId) => {
    setView(next);
    if (next === shownRef.current) {
      if (exitTimer.current !== null) {
        window.clearTimeout(exitTimer.current);
        exitTimer.current = null;
      }
      setPaneLeaving(false);
      return;
    }
    setPaneLeaving(true);
    if (exitTimer.current !== null) window.clearTimeout(exitTimer.current);
    exitTimer.current = window.setTimeout(() => {
      exitTimer.current = null;
      shownRef.current = next;
      setShown(next);
      setPaneLeaving(false);
      // a section change should start at its own top, not inherit the scroll
      // offset of the section we just left
      document.scrollingElement?.scrollTo({ top: 0 });
    }, VIEW_EXIT_MS);
  }, []);

  useEffect(
    () => () => {
      if (exitTimer.current !== null) window.clearTimeout(exitTimer.current);
    },
    [],
  );

  // custom model library (round-6): registry + url-download job + inputs
  const [customLib, setCustomLib] = useState<CustomListPayload | null>(null);
  const [customJob, setCustomJob] = useState<CustomJob | null>(null);
  const [importDir, setImportDir] = useState("");
  const [importMode, setImportMode] = useState<"link" | "copy">("link");
  const [importName, setImportName] = useState("");
  const [importBusy, setImportBusy] = useState(false);
  const [urlInput, setUrlInput] = useState("");
  const [urlName, setUrlName] = useState("");
  const [importNotice, setImportNotice] = useState<string | null>(null);
  // two-step mirror prompt: null = hidden, else the url awaiting the choice
  const [mirrorAsk, setMirrorAsk] = useState<{ url: string; hint: string } | null>(null);
  const [devUnlocked, setDevUnlocked] = useState(() => loadDevUnlocked());

  // custom-dl events carry a full job snapshot - just store it (event
  // phases start/progress both mean "job is running")
  const setCustomJobFromEvent = (ev: Extract<DlEvent, { kind: "custom-dl" }>) => {
    setCustomJob({
      phase: ev.phase === "progress" || ev.phase === "start" ? "running" : ev.phase,
      name: ev.name,
      url: ev.url,
      file: ev.file,
      done: ev.done,
      total: ev.total,
      // without this the "断点续传自 N" hint vanished on the first event
      resumed: ev.resumed,
      note: ev.note,
    });
  };

  const changeHlMode = (m: HighlightMode) => {
    setHlMode(m);
    saveHighlightMode(m);
  };

  const changeGlassPref = (p: GlassPreference) => {
    setGlassPref(p);
    saveGlassPref(p);
  };

  const notify = useCallback((msg: string) => setToast(msg), []);

  // round-30: notice stack helpers. Cap at three so a burst of unloads cannot
  // cover the results panel; the oldest card leaves through its own animation.
  const pushNotice = useCallback((n: Omit<NoticeItem, "id">) => {
    noticeSeq.current += 1;
    const item: NoticeItem = { ...n, id: noticeSeq.current };
    setNotices((list) => [...list.slice(-2), item]);
  }, []);

  const dismissNotice = useCallback((id: number) => {
    setNotices((list) => list.filter((x) => x.id !== id));
  }, []);

  useEffect(() => {
    if (!toast) return;
    const t = window.setTimeout(() => setToast(null), 3600);
    return () => window.clearTimeout(t);
  }, [toast]);

  // Right-click menu (2026-11-21): the WebView2 native menu cannot be styled,
  // so every contextmenu is replaced by the app's own rounded, spring-animated
  // menu. Items are enabled from the real target state.
  useEffect(() => {
    const onCtx = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest(".ctx-menu")) return;
      const field = textFieldOf(target);
      const sel = hasSelectionIn(field);
      e.preventDefault();
      setCtxMenu({
        seq: Date.now(),
        x: e.clientX,
        y: e.clientY,
        target: field,
        items: [
          {
            id: "cut",
            label: "剪切",
            shortcut: "Ctrl+X",
            disabled: !field || !sel,
            run: async () => {
              if (!(await cutSelection(field))) notify("剪切没有成功——请用 Ctrl+X");
            },
          },
          {
            id: "copy",
            label: "复制",
            shortcut: "Ctrl+C",
            disabled: !sel,
            run: async () => {
              if (!(await copySelection(field))) notify("复制没有成功——请用 Ctrl+C");
            },
          },
          {
            id: "paste",
            label: "粘贴",
            shortcut: "Ctrl+V",
            disabled: !field,
            run: async () => {
              const outcome = await pasteInto(field);
              if (outcome === "failed") {
                notify("读不到系统剪贴板——请用 Ctrl+V 重试");
              } else if (outcome === "empty") {
                // no text in the clipboard: keep the reason, the detect click
                // turns it into a readable explanation
                setPasteIssue("nontext");
                notify("剪贴板里是图片或文件，没有可粘贴的文字");
              }
            },
          },
          {
            id: "all",
            label: "全选",
            shortcut: "Ctrl+A",
            divider: true,
            run: () => selectAllIn(field),
          },
        ],
      });
    };
    document.addEventListener("contextmenu", onCtx);
    return () => document.removeEventListener("contextmenu", onCtx);
  }, [notify]);

  const applyRetention = async (tier: RetentionTier) => {
    setRetention(tier);
    retentionRef.current = tier;
    try {
      localStorage.setItem(RETENTION_KEY, tier);
    } catch {
      /* non-fatal */
    }
    const c = clientRef.current;
    if (!c) return;
    try {
      await c.request("resident/policy", { retention: tier });
      await refresh(c);
    } catch (e) {
      setError(String(e));
    }
  };

  const releaseAllResident = async () => {
    const c = clientRef.current;
    if (!c) return;
    try {
      await c.request("resident/unload", {});
      await refresh(c);
      notify("已释放全部驻留模型");
    } catch (e) {
      setError(String(e));
    }
  };

  // round-30 user spec: one × per resident capsule frees exactly that model.
  // The backend already accepts {engine_id}; the reply carries the post-unload
  // VRAM block, so "released N MB" is measured, not guessed.
  const unloadOne = async (r: ResidentEntry) => {
    const c = clientRef.current;
    if (!c || unloading) return;
    const before = catalog?.vram_used_mb ?? 0;
    setUnloading(r.id);
    try {
      const res = await c.request<UnloadPayload>("resident/unload", { engine_id: r.id });
      await refresh(c);
      const after = typeof res.vram_used_mb === "number" ? res.vram_used_mb : before;
      const freed = Math.max(0, before - after);
      const size = freed > 0 ? ` · 释放 ${fmtMb(freed)}` : r.actual_mb ? ` · ${fmtMb(r.actual_mb)}` : "";
      pushNotice({ kind: "ok", title: "已从显存卸载", body: `${r.name}${size}` });
    } catch (e) {
      pushNotice({ kind: "error", title: "卸载失败", body: `${r.name}：${String(e)}` });
    } finally {
      setUnloading(null);
    }
  };

  const clearPartials = useCallback(() => {
    if (partialFlush.current !== null) {
      window.cancelAnimationFrame(partialFlush.current);
      partialFlush.current = null;
    }
    partialsRef.current = {};
    setPartials({});
  }, []);

  // D29 batches arrive much faster than a human can read. Keep the stream
  // visible, but fold all batches into one animation frame instead of copying
  // an ever-growing array for every sentence group.
  const queuePartial = useCallback((engineId: string, segments: DetectSegment[]) => {
    if (segments.length === 0) return;
    const current = partialsRef.current[engineId] ?? [];
    current.push(...segments);
    partialsRef.current[engineId] = current;
    if (partialFlush.current !== null) return;
    partialFlush.current = window.requestAnimationFrame(() => {
      partialFlush.current = null;
      setPartials({ ...partialsRef.current });
    });
  }, []);

  useEffect(
    () => () => {
      if (partialFlush.current !== null) {
        window.cancelAnimationFrame(partialFlush.current);
        partialFlush.current = null;
      }
    },
    [],
  );

  const refresh = useCallback(async (c: SidecarClient) => {
    const [h, e, d] = await Promise.all([
      c.request<HealthPayload>("health"),
      c.request<EnginesPayload>("engines"),
      c.request<DlStatusPayload>("download/status"),
    ]);
    setHealth(h);
    setCatalog(e);
    setDl(d);
    // Diagnostics must not gate the main workbench if an older sidecar or one
    // damaged runtime component still answers health/engines.
    void c
      .request<RuntimeStatusPayload>("runtime/status")
      .then(setRuntimeStatus)
      .catch(() => {});
    void c.request<CustomListPayload>("custom/list").then(setCustomLib).catch(() => {});
    // dev-options raw-frame viewer (best-effort, never breaks the flow)
    try {
      (window as unknown as { __lastFrames?: unknown }).__lastFrames = {
        health: h,
        engines: e,
        download_status: d,
      };
    } catch {
      /* ignore */
    }
    // Engine picks: restore the saved selection on first catalogue arrival,
    // else default to every non-router engine so "detect" is always runnable.
    // Re-runs of refresh (reconnect/refresh click) never override the user's
    // in-session edits.
    setDetectEngines((prev) => {
      if (prev.size > 0) return prev;
      const saved = loadSavedEnginePick().filter((id) =>
        e.engines.some((x) => x.id === id),
      );
      if (saved.length > 0) return new Set(saved);
      // default to every installed non-router engine (dl arrived in the same
      // refresh; unknown install state -> lenient pick, the sanitize effect
      // below prunes once dl is known)
      return new Set(
        e.engines
          .filter(
            (x) =>
              x.kind !== "router" &&
              (d ? d.engines[x.id]?.installed === true : true),
          )
          .map((x) => x.id),
      );
    });
  }, []);

  const refreshRuntime = useCallback(async () => {
    const c = clientRef.current;
    if (!c) throw new Error("not connected");
    setRuntimeStatus(
      await c.request<RuntimeStatusPayload>("runtime/status", { refresh: true }),
    );
  }, []);

  // keep picks runnable: once the download status arrives, drop anything
  // that cannot actually run (stale saved picks, defaults picked before
  // dl was known). User edits are safe - chips refuse unusable engines.
  useEffect(() => {
    if (!dl || !catalog) return;
    setDetectEngines((prev) => {
      const usable = usableEngineIds(catalog, dl);
      const next = new Set([...prev].filter((id) => usable.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [dl, catalog]);

  // the model capsule must never point at an engine the last run did not run
  useEffect(() => {
    if (modelFilter === "all" || !detectResult) return;
    if (!detectResult.results.some((r) => r.engine === modelFilter)) setModelFilter("all");
  }, [detectResult, modelFilter]);

  const onEvent = async (c: SidecarClient, ev: DlEvent) => {
    if (ev.kind === "resident") {
      // idle sweep / manual unload changed the residency numbers server-side
      await refresh(c);
      return;
    }
    if (ev.kind === "custom-dl") {
      // custom library job: snapshot lives in the event itself
      setCustomJobFromEvent(ev);
      if (ev.phase === "done" || ev.phase === "error" || ev.phase === "cancelled") {
        void c.request<CustomListPayload>("custom/list").then(setCustomLib).catch(() => {});
      }
      return;
    }
    if (ev.kind === "detect_part") {
      // D29 streaming: batch into one React commit per frame.
      queuePartial(ev.engine_id, ev.segments);
      return;
    }
    if (ev.kind === "download_progress") {
      setProgress((p) => ({ ...p, [ev.engine_id]: { done: ev.done_bytes, total: ev.total_bytes, dest: ev.dest } }));
      return;
    }
    if (ev.kind === "cast_start") {
      setJobLog((l) => [...l.slice(-20), `▶ cast ${ev.engine_id} → ${ev.mode}`]);
      return;
    }
    if (ev.kind === "cast_progress") {
      setCastProgress((p) => ({
        ...p,
        [ev.engine_id]: { done: ev.done_bytes, total: ev.total_bytes, dest: ev.dest },
      }));
      return;
    }
    if (ev.kind === "cast_done") {
      setJobLog((l) => [...l.slice(-20), `✔ cast ${ev.engine_id} done (${ev.mode})`]);
      setCastProgress((p) => {
        const { [ev.engine_id]: _drop, ...rest } = p;
        return rest;
      });
    } else if (ev.kind === "cast_cancelled") {
      setJobLog((l) => [...l.slice(-20), `⏹ cast ${ev.engine_id} cancelled`]);
      setCastProgress((p) => {
        const { [ev.engine_id]: _drop, ...rest } = p;
        return rest;
      });
    } else if (ev.kind === "cast_error") {
      setJobLog((l) => [...l.slice(-20), `✗ cast ${ev.engine_id}: ${ev.error}`]);
      setCastProgress((p) => {
        const { [ev.engine_id]: _drop, ...rest } = p;
        return rest;
      });
    } else if (ev.kind === "download_engine_start") {
      const host = ev.source === "mirror" ? "hf-mirror.com" : "huggingface.co";
      setJobLog((l) => [...l.slice(-20), `▶ ${ev.engine_id}: ${ev.files} files · 源 ${host}`]);
      return;
    } else if (ev.kind === "download_file_done") {
      setJobLog((l) => [...l.slice(-20), `✓ ${ev.engine_id}/${ev.dest}`]);
      return;
    } else if (ev.kind === "download_engine_done") {
      setJobLog((l) => [...l.slice(-20), `✔ ${ev.engine_id} done`]);
      setProgress((p) => {
        const { [ev.engine_id]: _drop, ...rest } = p;
        return rest;
      });
    } else if (ev.kind === "download_cancelled") {
      setJobLog((l) => [...l.slice(-20), `⏹ ${ev.engine_id} cancelled`]);
      setProgress((p) => {
        const { [ev.engine_id]: _drop, ...rest } = p;
        return rest;
      });
    } else if (ev.kind === "download_error") {
      setJobLog((l) => [...l.slice(-20), `✗ ${ev.engine_id}: ${ev.error}`]);
      setProgress((p) => {
        const { [ev.engine_id]: _drop, ...rest } = p;
        return rest;
      });
    }
    // Only terminal task events need a three-request status refresh. File
    // progress and start events are already represented by their stream.
    await refresh(c); // installed/storage flags may have changed
  };

  const connect = (p?: string, t?: string) => {
    setError(null);
    // one page owns one socket: never leave the previous client's socket
    // open behind the new one (the sidecar allows a single client)
    clientRef.current?.disconnect();
    const c = new SidecarClient({ port: Number(p ?? port), token: t ?? token });
    clientRef.current = c;
    c.onState(setState);
    c.onEvent((ev) => void onEvent(c, ev));
    c.connect();
  };

  const disconnect = () => {
    clientRef.current?.disconnect();
    clientRef.current = null;
    setState("idle");
  };

  useEffect(() => {
    if (state === "open" && clientRef.current) {
      const c = clientRef.current;
      // the stored residency tier is authoritative for this window: apply it
      // before the first catalogue read so the budget is already correct
      c.request("resident/policy", { retention: retentionRef.current })
        .catch(() => {})
        .finally(() => refresh(c).catch((e) => setError(String(e))));
    }
  }, [state, refresh]);

  // Round-20: a run can outlive the socket that asked for it (grace window).
  // Its reply is parked in the sidecar and the page that took over fetches it
  // here. The parked text comes back with it so the highlight view has
  // sentences to draw - a pure-text run would otherwise be scores with
  // nothing underneath. This page never edited anything (the reload emptied
  // the box), so the restored input becomes the baseline, not a stale one.
  const adoptParkedResult = useCallback(
    async (c: SidecarClient, expectedJobId?: number) => {
      try {
        const res = await c.request<DetectPayload | { cancelled: true }>(
          "detect/result",
          expectedJobId === undefined ? {} : { job_id: expectedJobId },
        );
        if ("cancelled" in res) {
          // the run died before it produced anything (e.g. stopped from the
          // orphan strip right before the finish line)
          notify("这一轮在出结果前就被取消了");
          return;
        }
        const restored = res.input?.text ?? "";
        const ids = attachmentsRef.current.map((a) => a.id).join(",");
        setDetectResult(res);
        setModelFilter("all");
        setCompDetail(false);
        clearPartials();
        if (restored) setText(restored);
        setTextAtDetect(restored);
        setInputAtDetect({ text: restored, attachments: ids });
        notify(`已接管界面重载前发起的那一轮检测（${res.results.length} 个引擎）`);
      } catch (e) {
        setError(String(e));
      } finally {
        setJobNotice(null);
      }
    },
    [clearPartials, notify],
  );

  // Round-19: this page may not be the one that started the current run. Ask
  // the sidecar what the last job is doing - a reload kills the socket that
  // owned the reply, and the GPU work must not disappear silently.
  useEffect(() => {
    if (state !== "open" || !clientRef.current) return;
    const c = clientRef.current;
    let cancelled = false;
    void c
      .request<{ job: DetectJobInfo }>("detect/status")
      .then(({ job }) => {
        if (cancelled) return;
        if (job.state === "running") {
          setJobNotice(job);
          return;
        }
        if (job.has_result) {
          // the run finished while no page was listening; its answer waited
          void adoptParkedResult(c, job.job_id);
          return;
        }
        // Only a fresh "cancelled because the window went away" is worth a
        // toast; older history belongs in the log, not on every reconnect.
        const fresh = job.finished_at !== null && Date.now() / 1000 - job.finished_at < 120;
        if (job.state === "cancelled" && job.reason === "disconnect" && fresh) {
          setJobNotice(null);
          notify(
            `上一轮检测已随界面重载取消（${job.engines.length} 个引擎，跑了 ${Math.round(job.elapsed_s)} 秒）`,
          );
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [state, notify, adoptParkedResult]);

  // Poll only while an orphaned run is actually alive, so a finished job
  // clears the strip by itself instead of leaving a stale warning on screen.
  useEffect(() => {
    if (!jobNotice || jobNotice.state !== "running") return;
    const c = clientRef.current;
    if (!c) return;
    const timer = window.setInterval(() => {
      // a run this page owns is not an orphan: without this the strip would
      // reappear one tick after the user starts a fresh detect
      if (detectingRef.current) return;
      void c
        .request<{ job: DetectJobInfo }>("detect/status")
        .then(({ job }) => {
          if (job.state === "running") {
            setJobNotice(job);
            return;
          }
          setJobNotice(null);
          if (job.has_result) {
            // the grace window kept the run alive and it just finished
            void adoptParkedResult(c, job.job_id);
            return;
          }
          if (job.state === "cancelled" && job.reason === "disconnect") {
            // the reaper won: no page took over inside the grace window
            notify("上一轮检测已取消：界面重载后没有页面及时接管它");
          }
        })
        .catch(() => {});
    }, 2000);
    return () => window.clearInterval(timer);
  }, [jobNotice, adoptParkedResult, notify]);

  // Startup gate clock: counts from the last time the work pane had no
  // catalogue, so a reconnect restarts the wait instead of inheriting it.
  const ready = state === "open" && catalog !== null;
  useEffect(() => {
    if (ready) {
      setBootSeconds(0);
      return;
    }
    const started = Date.now();
    setBootSeconds(0);
    const timer = window.setInterval(
      () => setBootSeconds(Math.floor((Date.now() - started) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [ready, state]);

  const retryConnect = async () => {
    setError(null);
    // inside the shell the port/token come from Rust: re-ask instead of reusing
    // whatever the first (possibly failed) handshake left in state
    let p = port;
    let t = token;
    if (inTauri()) {
      try {
        const info = await invoke<{ port: number; token: string }>("sidecar_info");
        p = String(info.port);
        t = info.token;
        setPort(p);
        setToken(t);
      } catch (e) {
        const detail = String(e);
        if (/找不到|失败|不完整|退出/.test(detail)) setError(detail);
        /* sidecar still not ready - fall through and try the known values */
      }
    }
    clientRef.current?.disconnect();
    // the old socket needs a beat to be reaped before the single-client rule
    // lets the new one in - do not race it
    window.setTimeout(() => connect(p, t), 400);
  };

  const cancelOrphanRun = async () => {
    const c = clientRef.current;
    if (!c) return;
    try {
      const res = await c.request<{ cancel_requested: boolean; job: DetectJobInfo }>(
        "detect/cancel",
      );
      if (res.job.has_result) {
        // it crossed the finish line just as the user hit stop: an answer
        // already exists, and throwing it away would be the rudest outcome
      await adoptParkedResult(c, res.job.job_id);
        return;
      }
      setJobNotice(res.job.state === "running" ? res.job : null);
      if (res.cancel_requested) notify("已请求取消——当前批次跑完就停");
    } catch (e) {
      setError(String(e));
    }
  };

  // entering the models view (re)fetches the custom library snapshot
  useEffect(() => {
    if (view === "models" && state === "open" && clientRef.current) {
      void refreshCustom(clientRef.current);
    }
  }, [view, state]);

  // Inside the Tauri shell: fetch {port, token} from Rust and connect
  // automatically (handshake may lag the window - retry briefly).
  useEffect(() => {
    if (!("__TAURI_INTERNALS__" in window)) return;
    let cancelled = false;
    (async () => {
      let lastHandshakeError = "";
      for (let i = 0; i < 20; i++) {
        try {
          const info = await invoke<{ port: number; token: string }>("sidecar_info");
          if (cancelled) return;
          setPort(String(info.port));
          setToken(info.token);
          connect(String(info.port), info.token);
          return;
        } catch (e) {
          lastHandshakeError = String(e);
          // A missing/corrupt runtime is actionable immediately; a normal
          // Python cold start is not, so only surface explicit launch errors.
          if (/找不到|失败|不完整|退出/.test(lastHandshakeError)) {
            setError(lastHandshakeError);
          }
          await new Promise((r) => setTimeout(r, 500));
        }
      }
      if (!cancelled) {
        setError(
          lastHandshakeError && lastHandshakeError !== "sidecar handshake not ready"
            ? lastHandshakeError
            : "sidecar handshake 超时（Rust 侧未就绪）",
        );
      }
    })();
    return () => {
      cancelled = true;
      // HMR re-runs this effect; without the disconnect the old socket stayed
      // OPEN and kept the sidecar's single-client slot, so every reconnect
      // was refused (round-28). The next setup creates a fresh client.
      clientRef.current?.disconnect();
      clientRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleCheck = (eid: string, risk: string, riskTextKey: string | null) => {
    if (checked.has(eid)) {
      setChecked((prev) => {
        const next = new Set(prev);
        next.delete(eid);
        return next;
      });
      return;
    }
    if (RISK_DUAL_DIALOG.has(risk)) {
      setGate({ engineId: eid, riskLevel: risk, riskTextKey }); // D20 gate
      return;
    }
    setChecked((prev) => new Set(prev).add(eid));
  };

  const startDownload = async () => {
    const c = clientRef.current;
    if (!c || checked.size === 0) return;
    setError(null);
    try {
      const res = await c.request<{ started: string[]; skipped: { engine_id: string; reason: string }[] }>(
        "download/start",
        { engines: [...checked], source: downloadSource },
      );
      const lines = [
        `源: ${downloadSource === "mirror" ? "hf-mirror.com" : "huggingface.co"}（设置 → 常规）`,
        ...res.started.map((s) => `started: ${s}`),
        ...res.skipped.map((s) => `skipped: ${s.engine_id} (${s.reason})`),
      ];
      setJobLog((l) => [...l.slice(-20), ...lines]);
      await refresh(c);
    } catch (e) {
      setError(String(e));
    }
  };

  const cancelDownload = async () => {
    const c = clientRef.current;
    if (!c) return;
    try {
      await c.request("download/cancel", {});
    } catch (e) {
      setError(String(e));
    }
  };

  // ---- custom model library (round-6): import + URL download -------------

  const refreshCustom = async (c: SidecarClient) => {
    try {
      setCustomLib(await c.request<CustomListPayload>("custom/list"));
    } catch {
      /* not connected / mid-reconnect: leave the last snapshot */
    }
  };

  const runImport = async () => {
    const c = clientRef.current;
    if (!c || !importDir.trim()) return;
    setImportBusy(true);
    setError(null);
    try {
      const res = await c.request<CustomImportResult>("custom/import", {
        source_dir: importDir.trim(),
        mode: importMode,
        name: importName.trim() || undefined,
      });
      await refreshCustom(c);
      // an engine import wrote into models/<dest>: re-read health/engines/
      // download-status too, otherwise the workbench still treats it as
      // un-installed until the next reconnect (the note already promised
      // "立即可用").
      if (res.kind === "engine") await refresh(c);
      setImportNotice(
        res.kind === "engine"
          ? res.verify?.installed
            ? `导入完成：已按官方引擎「${engineName(res.engine ?? "")}」就位，检测立即可用。`
            : `导入完成，但文件校验未通过——请核对来源是否完整。${res.note}`
          : `导入完成：${res.note}（未匹配官方引擎，登记为库存，引擎接入见后续版本）`,
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setImportBusy(false);
    }
  };

  // URL download step 1: mirror probe -> friendly two-step prompt.
  // Host list mirrors backend custom_repo._MIRRORS.
  const askUrlDownload = () => {
    const url = urlInput.trim();
    if (!/^https?:\/\//i.test(url)) {
      setError("URL 必须以 http:// 或 https:// 开头");
      return;
    }
    let host = "";
    try {
      host = new URL(url).hostname;
    } catch {
      /* unparseable: fall through to direct hint */
    }
    const known = MIRROR_HOSTS.some((h) => host === h || host.endsWith("." + h));
    setMirrorAsk({
      url,
      hint: known
        ? "检测到 Hugging Face 源——已有公共镜像站 hf-mirror.com。是否尝试使用镜像站下载？"
        : "该源没有已知镜像站，将直连下载。",
    });
  };

  const runUrlDownload = async (mirror: "mirror" | "direct") => {
    const c = clientRef.current;
    if (!c || !mirrorAsk) return;
    setMirrorAsk(null);
    setError(null);
    try {
      const job = await c.request<CustomJob>("custom/download", {
        url: mirrorAsk.url,
        mirror,
        name: urlName.trim() || undefined,
      });
      setCustomJob(job);
    } catch (e) {
      setError(String(e));
    }
  };

  const cancelCustomJob = async () => {
    const c = clientRef.current;
    if (!c) return;
    try {
      await c.request("custom/cancel", {});
    } catch (e) {
      setError(String(e));
    }
  };

  // engines for this run (round-6): custom -> manual picks; the other modes
  // resolve the profile set from the text language (or the confirmed precise
  // language) and the hardware tier. Both paths are filtered to engines that
  // can actually run (installed, not the router) so the "（N 引擎）" count
  // always matches what detect will really execute.
  //
  // round-32: the same resolver now serves the capsule's per-language preview,
  // so the two can never disagree about what a language would run.
  const enginesFor = useCallback(
    (m: DetectMode, lang: DetectLang): string[] => {
      const usable = usableEngineIds(catalog, dl);
      if (m === "custom") return [...detectEngines].filter((id) => usable.has(id));
      const ids = new Set(catalog?.engines.map((e) => e.id) ?? []);
      return resolveModeEngines(m, lang, health?.hw?.tier, ids, usable);
    },
    [catalog, dl, detectEngines, health],
  );

  // round-34: an empty box has no detection language. `autoLang === null`
  // means "not decided yet" — the capsule says so instead of asserting 中文,
  // and the run set falls back to the zh profile only as a display default
  // (the run button is disabled while the box is empty anyway).
  const autoLang: DetectLang | null = detectLangOrNull(text);
  const activeLang: DetectLang | null = mode === "precise" ? preciseLang : autoLang;
  const runLang: DetectLang = activeLang ?? "zh";
  const runEngines = useMemo(() => enginesFor(mode, runLang), [enginesFor, mode, runLang]);
  // preview follows the real language again whenever the input switches
  useEffect(() => setPreviewLang(runLang), [runLang]);
  const previewEngines = useMemo(
    () => (previewLang === runLang ? runEngines : enginesFor(mode, previewLang)),
    [previewLang, runLang, runEngines, enginesFor, mode],
  );
  const langSets = useMemo(
    () =>
      DETECT_LANGS.map((l) => ({
        ...l,
        ids: l.id === runLang ? runEngines : enginesFor(mode, l.id),
      })),
    [enginesFor, mode, runLang, runEngines],
  );

  // friendly engine display name (round-6: no raw ids in the run UI)
  const engineName = (id: string): string =>
    catalog?.engines.find((e) => e.id === id)?.name || id;

  const commitMode = (m: DetectMode, pl: "zh" | "en") => {
    setMode(m);
    setPreciseLang(pl);
    try {
      localStorage.setItem(MODE_KEY, m);
      localStorage.setItem(PRECISE_LANG_KEY, pl);
    } catch {
      /* ignore */
    }
  };

  const togglePick = (id: string) => {
    setDetectEngines((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      saveEnginePick(next);
      return next;
    });
  };

  // bulk replace (custom sheet: 全选 / 清空 / 反选) — one state write and one
  // localStorage write instead of N toggles, so a 9-engine clear is instant
  const replacePicks = (ids: string[]) => {
    const next = new Set(ids);
    setDetectEngines(next);
    saveEnginePick(next);
  };

  // Ctrl+V is handled by the webview itself, so the only place we can see what
  // the clipboard actually held is the DOM paste event: its DataTransfer lists
  // the flavours without any permission check. Nothing textual -> remember the
  // reason instead of letting the paste silently do nothing.
  const onPasteText = useCallback((e: ReactClipboardEvent<HTMLTextAreaElement>) => {
    const dt = e.clipboardData;
    if (!dt) return;
    const types = Array.from(dt.types ?? []);
    // any text flavour (plain or html) is the editor's business, not ours: Word
    // and browsers often put only text/html and still paste fine
    if (dt.getData("text/plain").trim() || types.some((t) => t.startsWith("text/"))) {
      setPasteIssue(null);
      return;
    }
    if (types.length === 0) return; // empty clipboard: nothing to explain
    e.preventDefault();
    setPasteIssue(
      types.includes("Files")
        ? "file"
        : types.some((t) => t.startsWith("image/"))
          ? "image"
          : "nontext",
    );
  }, []);

  // acknowledging the notice clears the flag: with an empty box and no pending
  // paste there is nothing left to detect, so the button goes back to disabled
  const dismissFormat = useCallback(() => {
    setFormatOpen(false);
    setPasteIssue(null);
  }, []);

  // ---- D40 file import ---------------------------------------------------
  const submitImport = useCallback(
    async (payload: { paths: string[] } | { blobs: FileBlob[] }) => {
      const c = clientRef.current;
      if (!c) {
        setAttErrors([{ name: "", error: "还没连上 sidecar" }]);
        return;
      }
      setAttBusy(true);
      setAttErrors([]);
      try {
        const res = await c.request<FileImportPayload>("file/import", payload);
        setAttachments(res.all);
        if (res.errors.length) setAttErrors(res.errors);
      } catch (e) {
        setAttErrors([{ name: "", error: String(e) }]);
      } finally {
        setAttBusy(false);
      }
    },
    [],
  );

  // One confirmation, on the first import into a non-empty text box. After that
  // the attachments simply append — the composed document is text-first, then
  // cards in order, so nothing is destroyed without the user saying so.
  const askThenImport = useCallback(
    (action: () => void) => {
      const typed = text.trim();
      if (typed && attachments.length === 0) {
        setImportAsk({
          chars: typed.length,
          run: (keepText) => {
            if (!keepText) setText("");
            action();
          },
        });
        return;
      }
      action();
    },
    [text, attachments.length],
  );

  // The confirmation decision belongs to the *entry* point (button / drop), not
  // here: by the time the bytes exist the user has already answered, and asking
  // again would fire a second modal for a single import.
  const acceptBrowserFiles = useCallback(
    async (files: File[]) => {
      if (!files.length) return;
      const tooBig = oversized(files);
      if (tooBig.length) {
        setAttErrors(tooBig.map((n) => ({ name: n, error: "文件超过 20 MB，未导入" })));
      }
      const ok = files.filter((f) => f.size <= MAX_BYTES);
      if (!ok.length) return;
      await submitImport({ blobs: await toBlobs(ok) });
    },
    [submitImport],
  );

  const pickAndImport = useCallback(() => {
    askThenImport(() => {
      void (async () => {
        if (inTauri()) {
          try {
            const paths = await pickPaths();
            if (paths.length) await submitImport({ paths });
          } catch (e) {
            setAttErrors([{ name: "", error: String(e) }]);
          }
          return;
        }
        const files = await pickBrowserFiles();
        await acceptBrowserFiles(files);
      })();
    });
  }, [askThenImport, submitImport, acceptBrowserFiles]);

  const acceptDroppedPaths = useCallback(
    (paths: string[]) => {
      if (!paths.length) return;
      askThenImport(() => {
        void submitImport({ paths });
      });
    },
    [askThenImport, submitImport],
  );

  const removeAttachment = useCallback(
    async (id: string) => {
      const c = clientRef.current;
      if (!c) return;
      // optimistic: the card disappears at once, the list is re-synced after
      setAttachments((list) => list.filter((a) => a.id !== id));
      try {
        const res = await c.request<{ dropped: boolean; all: AttachmentMeta[] }>("file/drop", { id });
        setAttachments(res.all);
      } catch (e) {
        setAttErrors([{ name: "", error: String(e) }]);
      }
    },
    [],
  );

  const reorderAttachments = useCallback(
    (ids: string[]) => {
      setAttachments((list) => {
        const byId = new Map(list.map((a) => [a.id, a]));
        return ids.map((id) => byId.get(id)).filter((a): a is AttachmentMeta => Boolean(a));
      });
    },
    [],
  );

  // Native drag-and-drop inside the shell: the webview swallows HTML5 file
  // drops, so the Tauri event is the only route that yields real paths.
  // Positions arrive in physical pixels.
  const dropHandlerRef = useRef<(paths: string[], inside: boolean) => void>(() => {});
  dropHandlerRef.current = (paths, inside) => {
    if (inside) acceptDroppedPaths(paths);
  };

  useEffect(() => {
    if (!inTauri()) return;
    let dispose: (() => void) | null = null;
    let cancelled = false;
    void (async () => {
      const { getCurrentWebview } = await import("@tauri-apps/api/webview");
      const un = await getCurrentWebview().onDragDropEvent((event) => {
        const p = event.payload;
        const hit = (pos: { x: number; y: number }) => {
          const el = composerRef.current;
          if (!el) return false;
          const r = el.getBoundingClientRect();
          const dpr = window.devicePixelRatio || 1;
          const x = pos.x / dpr;
          const y = pos.y / dpr;
          return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
        };
        if (p.type === "over") setDropHot(hit(p.position));
        else if (p.type === "leave") setDropHot(false);
        else if (p.type === "drop") {
          setDropHot(false);
          dropHandlerRef.current(p.paths, hit(p.position));
        }
      });
      if (cancelled) un();
      else dispose = un;
    })();
    return () => {
      cancelled = true;
      dispose?.();
    };
  }, []);

  // D40: the table lives in the sidecar, which outlives a page reload (it only
  // dies with the app window). Re-sync on connect instead of assuming empty,
  // otherwise the UI would show no cards while the sidecar still holds text.
  useEffect(() => {
    if (state !== "open") return;
    const c = clientRef.current;
    if (!c) return;
    void c
      .request<{ items: AttachmentMeta[] }>("file/list", {})
      .then((res) => setAttachments(res.items))
      .catch(() => {
        /* the UI can still work; a failed sync just leaves the list as-is */
      });
  }, [state]);

  const runDetect = async () => {
    const c = clientRef.current;
    if (!text.trim() && attachments.length === 0) {
      // an image/file paste leaves the box empty: say why rather than no-op
      if (pasteIssue) setFormatOpen(true);
      return;
    }
    const attachmentChars = attachments.reduce((sum, a) => sum + a.chars, 0);
    if (text.length + attachmentChars > MAX_INPUT_CHARS) {
      notify(
        `本次输入超过 ${MAX_INPUT_CHARS.toLocaleString("zh-CN")} 字上限，请删减文字或附件后再检测。`,
      );
      return;
    }
    if (!c || runEngines.length === 0) return;
    if (jobNotice && jobNotice.state === "running") {
      // the sidecar would refuse this anyway ("already in progress"); saying
      // it here keeps the orphan strip on screen and names the way out
      notify("上一轮检测还在跑，先等它结束或点「取消这一轮」");
      return;
    }
    setError(null);
    setDetecting(true);
    setJobNotice(null); // this page owns the slot now
    clearPartials(); // fresh stream
    setDetectResult(null);
    setModelFilter("all"); // a new run may not contain the previously shown engine
    setCompDetail(false);
    try {
      const res = await c.request<DetectPayload | { cancelled: true; job?: DetectJobInfo }>(
        "detect",
        {
          text,
          engines: runEngines,
          attachments: attachments.map((a) => a.id),
        },
      );
      if ("cancelled" in res) {
        // stopped server-side before it produced a result (round-19): there is
        // nothing to render, and pretending otherwise would blank the panel
        notify("这一轮在出结果前就被取消了");
        return;
      }
      setDetectResult(res);
      // With attachments the composed document (text + file bodies) only exists
      // in the sidecar; the reply hands it back once so the highlight view and
      // the per-sentence list can show the real sentences. Pure-text runs keep
      // the old contract: the raw text never travels back.
      setTextAtDetect(res.input?.text ?? text);
      setInputAtDetect({ text, attachments: attachments.map((a) => a.id).join(",") });
    } catch (e) {
      setError(String(e));
    } finally {
      clearPartials();
      setDetecting(false);
      // residency numbers changed (engines just loaded / may have been
      // released by the "none" tier): re-read instead of showing stale text
      if (clientRef.current) {
        void refresh(clientRef.current).catch(() => {});
      }
    }
  };

  // ---- derived view data (D28/D29) ---------------------------------------

  // D40: every result view (spans, warning, highlight, sentence list) must read
  // the text that was actually scored. With attachments that text only arrives
  // in the detect reply, so the results render from this snapshot while the
  // composer keeps whatever the user typed next.
  const attachmentIds = attachments.map((a) => a.id).join(",");
  const textDirty =
    detectResult !== null &&
    (text !== inputAtDetect.text || attachmentIds !== inputAtDetect.attachments);
  const resultText = detectResult !== null ? textAtDetect : text;
  const healthInfo = deriveHealth({
    conn: state,
    downloading: dl?.running === true,
    casting: Object.keys(castProgress).length > 0,
    detecting,
    error,
    textDirty,
    hasResult: detectResult !== null,
  });

  // live view: partials while detecting, full result once the reply lands
  const viewResults = useMemo<DetectResult[]>(() => {
    if (detecting && Object.keys(partials).length > 0) {
      return Object.entries(partials).map(([engine, segments]) => ({
        engine,
        status: "ok" as const,
        segments,
        doc_score: null,
        detail: undefined,
      }));
    }
    return detectResult?.results ?? [];
  }, [detecting, partials, detectResult]);

  const docLength = useMemo(() => codePointLength(resultText), [resultText]);
  const spans = useMemo(() => aggregateMax(viewResults, docLength), [viewResults, docLength]);
  const composite = useMemo(() => overallScore(spans), [spans]);
  const warning = useMemo(() => languageWarning(resultText, viewResults), [resultText, viewResults]);
  const okEngines = viewResults.filter((r) => r.status === "ok");
  const failedEngines = viewResults.filter((r) => r.status !== "ok");
  const dlEngines = useMemo(() => Object.entries(dl?.engines ?? {}), [dl]);

  // friendly names everywhere a result row is rendered (catalogue names, not ids)
  const engineNames = useMemo(() => {
    const m = new Map<string, string>();
    for (const e of catalog?.engines ?? []) m.set(e.id, e.name || e.id);
    return m;
  }, [catalog]);
  const nameOf = useCallback((id: string) => engineNames.get(id) ?? id, [engineNames]);

  // language tag for the engine rows (names carry (EN)/(CN) but the tag
  // states the detection type explicitly, same wording as the picker).
  const langLabel = useCallback((id: string) => {
    const lang = catalog?.engines.find((e) => e.id === id)?.lang ?? "";
    return ENGINE_LANG_TAG[lang] ?? (lang || "—");
  }, [catalog]);

  // Which scores feed the coloured document: "all" = per-sentence max,
  // an engine id = that engine alone (2026-11-21 model capsule).
  const modelSpans = useMemo<SentenceSpan[]>(() => {
    if (modelFilter === "all") return aggregateMax(viewResults, docLength);
    const r = viewResults.find((x) => x.engine === modelFilter && x.status === "ok");
    if (!r) return [];
    return r.segments
      .filter((s) => s.length < docLength) // doc-level segment: no sentence colour
      .map((s) => ({
        offset: s.offset,
        length: s.length,
        score: s.score,
        sources: [{ engine: r.engine, score: s.score }],
      }))
      .sort((a, b) => a.offset - b.offset);
  }, [modelFilter, viewResults, docLength]);

  // original document position for every span (kept in the row even when the
  // list is re-ordered by rate, so "第 12 句" still means the 12th sentence)
  const docIndex = useMemo(() => {
    const m = new Map<string, number>();
    [...modelSpans]
      .sort((a, b) => a.offset - b.offset)
      .forEach((s, i) => m.set(`${s.offset}:${s.length}`, i + 1));
    return m;
  }, [modelSpans]);

  // Filtering + ordering rules (user spec):
  //   只看 AI 率较高 -> descending; 只看比较像人工 -> most human first;
  //   全部句子 -> the 原文顺序/按 AI 率 switch decides.
  const sentenceRows = useMemo<SentenceRow[]>(() => {
    const filtered = modelSpans.filter((s) => passesFilter(s.score, sentFilter));
    const order: "text" | "desc" | "asc" =
      sentFilter === "ai"
        ? "desc"
        : sentFilter === "human"
          ? "asc"
          : sortPref === "rate"
            ? "desc"
            : "text";
    const sorted =
      order === "text"
        ? filtered
        : [...filtered].sort((a, b) => (order === "desc" ? b.score - a.score : a.score - b.score));
    return sorted.map((span) => ({
      span,
      index: docIndex.get(`${span.offset}:${span.length}`) ?? 0,
    }));
  }, [modelSpans, sentFilter, sortPref, docIndex]);

  const sentenceNote =
    sentFilter === "ai"
      ? "只看 AI 率较高 · 从高到低"
      : sentFilter === "human"
        ? "只看比较像人工 · 最像人工的在前"
        : sortPref === "rate"
          ? "全部句子 · 按 AI 率降序"
          : "全部句子 · 原文顺序";

  const modelOptions = useMemo(
    () => [
      { id: "all", label: "综合（全部引擎）", hint: "逐句取各引擎最高分" },
      ...okEngines.map((r) => ({
        id: r.engine,
        label: engineNames.get(r.engine) ?? r.engine,
        hint: r.doc_score === null ? "整篇 —" : `整篇 ${Math.round(r.doc_score * 100)}%`,
      })),
    ],
    [okEngines, engineNames],
  );

  const residentDetail = catalog?.resident_detail ?? [];
  const residentTitle = useMemo(() => {
    if (residentDetail.length === 0) return "当前没有模型驻留在显存里";
    return [
      "驻留明细（实测为本进程 torch 张量占用）：",
      ...residentDetail.map(
        (r) =>
          `${r.name} · ${r.actual_mb === null ? `${fmtMb(r.est_mb)}（估算）` : fmtMb(r.actual_mb)} · 已空闲 ${Math.round(r.idle_s / 60)} 分钟`,
      ),
    ].join("\n");
  }, [residentDetail]);

  // P3-b glass state (D28⑤/D30①): any heavy GPU/compositing work flattens
  // the glass surfaces for 300ms-easy swap; cpu tier stays flat in auto.
  const glassBusy =
    detecting || dl?.running === true || Object.keys(castProgress).length > 0;
  const glassState = resolveGlass(glassPref, health?.hw?.tier, glassBusy);

  return (
    <div className={"shell" + (inTauri() ? " in-tauri" : "")} data-glass={glassState}>
      <TitleBar />
      <aside className="sidebar glass">
        <div className="brand">
          {/* D38 tier C (<= 20px structure): at the 21px sidebar size the
              complete variant's 20-unit strokes render at 0.8px and turn grey */}
          <svg className="brand-mark" viewBox="0 0 512 512" aria-hidden="true" focusable="false">
            <rect
              x="2"
              y="2"
              width="508"
              height="508"
              rx="72"
              fill="#FFFFFF"
              stroke="#F2F2F2"
              strokeWidth="2"
            />
            <g fill="none" stroke="#000000" strokeLinecap="round" strokeLinejoin="round">
              <path
                d="M413 285 V60 Q413 35 388 35 H110 Q85 35 85 60 V412 Q85 437 110 437 H290"
                strokeWidth="32"
              />
              <path d="M147 160 H228 M147 270 H228" strokeWidth="38" />
              <circle cx="352" cy="362" r="74" fill="#FFFFFF" strokeWidth="30" />
              <path d="M410 420 L458 468" strokeWidth="32" />
            </g>
          </svg>
          <span>AIGC 检测台</span>
        </div>
        <HealthLight state={healthInfo.state} reason={healthInfo.reason} />
        <nav className="side-nav">
          <button
            className={view === "work" ? "nav-active" : ""}
            onClick={() => switchView("work")}
          >
            工作台
          </button>
          <button
            className={view === "models" ? "nav-active" : ""}
            onClick={() => switchView("models")}
          >
            模型管理
          </button>
          <button
            className={view === "runtime" ? "nav-active" : ""}
            onClick={() => switchView("runtime")}
          >
            基础服务组件
          </button>
          <button
            className={view === "settings" ? "nav-active" : ""}
            onClick={() => switchView("settings")}
          >
            设置
          </button>
        </nav>
        <div className="side-spacer" />
        <div className="side-foot">
          <DevicePanel hw={health?.hw} />
        </div>
      </aside>
      <main className="canvas">
        <div
          key={shown}
          className={"view-pane" + (paneLeaving ? " leaving" : "")}
        >
        {shown === "work" && (
        <div className="greeting">
          <span>{greetingText()}</span>
        </div>
        )}
        {shown === "work" && !catalog && (
          <StartupGate
            conn={state}
            seconds={bootSeconds}
            error={error}
            onRetry={retryConnect}
            onReload={() => window.location.reload()}
          />
        )}

        {/* the gate prints the same error inline - no double banner */}
        {error && !(shown === "work" && !catalog) && (
          <div className="error-box">
            <pre className="error">{error}</pre>
            <button onClick={() => setError(null)}>知道了</button>
          </div>
        )}

      {/* ---- protagonist: engines on top -> paste -> read ----------------- */}
      {shown === "work" && catalog && (
        <section className="detect-section" id="detect">
          <div
            className={
              "composer" + (composerTall ? " tall" : "") + (dropHot ? " drop-hot" : "")
            }
            ref={composerRef}
            onDragOver={(e) => {
              // Inside the shell the native dragDropEvent is the only source of
              // real paths, and HTML5 drops never fire; keep this branch for
              // the browser build / verification runs.
              if (inTauri()) return;
              if (!e.dataTransfer.types.includes("Files")) return;
              e.preventDefault();
              e.dataTransfer.dropEffect = "copy";
              if (!dropHot) setDropHot(true);
            }}
            onDragLeave={(e) => {
              if (inTauri()) return;
              if (e.currentTarget.contains(e.relatedTarget as Node | null)) return;
              setDropHot(false);
            }}
            onDrop={(e) => {
              if (inTauri()) return;
              const files = Array.from(e.dataTransfer.files);
              if (!files.length) return;
              e.preventDefault();
              setDropHot(false);
              askThenImport(() => void acceptBrowserFiles(files));
            }}
          >
            {/* engine strip (2026-11-21): mode capsule + this run's engines
                moved above the textarea, where the flow actually starts; the
                zoom control sits at the very top of the input window */}
            <div className="composer-strip">
              <button
                ref={capsuleRef}
                className="mode-capsule"
                onClick={() => setModeOpen((o) => !o)}
              >
                {MODE_LABELS[mode]}
                <span className="mode-caret">▾</span>
              </button>
              <InfoCapsule
                label={mode === "custom" ? "自定义引擎" : "引擎构成"}
                width={310}
              >
                {mode === "custom" ? (
                  <div className="pop-head pop-head-split">
                    <span>自定义勾选的引擎</span>
                    <span className="pop-head-note">{runEngines.length} 个</span>
                  </div>
                ) : (
                  <>
                    {/* round-34: neutral three-part statement — preset (head),
                        language (sub), engines (rows). The button itself no
                        longer claims a count: a per-language count on a single
                        capsule was exactly the confusion this replaces. */}
                    <div className="pop-head pop-head-split">
                      <span>各语言使用的引擎</span>
                      <span className="pop-head-note">{MODE_LABELS[mode]}预设</span>
                    </div>
                    {/* one tab per detection language, rendered from
                        DETECT_LANGS — adding a language there grows this row
                        automatically. The dot marks the group this input will
                        actually run; the other tabs are read-only previews. */}
                    <div className="lang-tabs" role="tablist" aria-label="检测语言">
                      {langSets.map((l) => (
                        <button
                          key={l.id}
                          type="button"
                          role="tab"
                          aria-selected={previewLang === l.id}
                          className={"lang-tab" + (previewLang === l.id ? " on" : "")}
                          onClick={() => setPreviewLang(l.id)}
                          title={
                            l.id === activeLang
                              ? "当前输入会使用这一组引擎"
                              : `查看${l.label}文本会使用的引擎（不影响本次检测）`
                          }
                        >
                          {l.label}
                          <span className="lang-count">{l.ids.length}</span>
                          {l.id === activeLang && (
                            <span className="lang-live" aria-label="本次使用" />
                          )}
                        </button>
                      ))}
                    </div>
                    <p className="pop-sub">
                      {langWithNoun(previewLang, "文本的引擎组合")}
                      {activeLang === null && "（未输入文本，暂按中文展示）"}
                    </p>
                  </>
                )}
                {previewEngines.map((id) => (
                  <div className="pop-row" key={id}>
                    <span className="pop-row-name">{nameOf(id)}</span>
                    <span className="pop-row-tag">{langLabel(id)}</span>
                  </div>
                ))}
                {previewEngines.length === 0 && (
                  <p className="pop-foot">
                    {mode === "custom"
                      ? "还没勾选引擎——点左侧模式胶囊进「自定义」逐个勾选。"
                      : `${detectLangLabel(previewLang)}暂无可用引擎——先下载模型，或换一个模式。`}
                  </p>
                )}
                <p className="pop-foot">
                  {mode === "custom"
                    ? "以上为自定义勾选的引擎，改勾选请点左侧模式胶囊。"
                    : activeLang === null
                      ? "检测语言待判定：输入文本后按字符自动判定，这里会换成对应语言的引擎。"
                      : `本次检测：${detectLangLabel(activeLang)} · ${runEngines.length} 个引擎（${
                          mode === "precise" ? "按你确认的文本语言" : "语言按输入文本自动判定"
                        }）。改选择请点左侧模式胶囊。`}
                </p>
              </InfoCapsule>
              <span className="strip-spacer" />
              <button
                type="button"
                className={"composer-zoom" + (composerTall ? " on" : "")}
                onClick={() => setComposerTall((v) => !v)}
                title={composerTall ? "还原输入框高度" : "放大输入框"}
                aria-label={composerTall ? "还原输入框高度" : "放大输入框"}
                aria-pressed={composerTall}
              >
                <svg
                  viewBox="0 0 16 16"
                  width="15"
                  height="15"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                >
                  <circle cx="7" cy="7" r="4.2" />
                  <path d="M10.2 10.2 14 14" />
                  <path d="M5 7h4M7 5v4" />
                </svg>
              </button>
            </div>
            {modeOpen && (
              <ModelModePopover
                mode={mode}
                preciseLang={preciseLang}
                catalog={catalog}
                tier={health?.hw?.tier}
                dl={dl}
                picks={detectEngines}
                customModels={customLib?.models ?? []}
                anchorRef={capsuleRef}
                onPickToggle={togglePick}
                onReplacePicks={replacePicks}
                onCommit={commitMode}
                onClose={() => setModeOpen(false)}
              />
            )}

            {/* D40: imported documents sit above the textarea as cards; the
                numbers on them are the compose order that detect sends */}
            <AttachmentCards
              items={attachments}
              onRemove={(id) => void removeAttachment(id)}
              onReorder={reorderAttachments}
              disabled={detecting || attBusy}
            />
            {attErrors.length > 0 && (
              <div className="att-errors" role="alert">
                {attErrors.map((e, i) => (
                  <div className="att-error" key={`${e.name}-${i}`}>
                    <span>
                      {e.name ? <b>{e.name}</b> : null}
                      {e.name ? "：" : ""}
                      {e.error}
                    </span>
                    <button
                      type="button"
                      aria-label="忽略这条提示"
                      onClick={() => setAttErrors((list) => list.filter((_, j) => j !== i))}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
            <textarea
              rows={5}
              value={text}
              maxLength={MAX_INPUT_CHARS}
              onChange={(e) => {
                setText(e.target.value);
                // typing (or a real text paste) makes the last failed paste moot
                if (e.target.value.trim()) setPasteIssue(null);
              }}
              onPaste={onPasteText}
              placeholder="把要检测的文章粘贴到这里，或把 docx / pdf / txt / md 文件拖进来…（支持中文/英文，长文逐段流式出结果）"
            />
            <div className="composer-row">
              <button
                className="primary"
                onClick={runDetect}
                disabled={
                  state !== "open" ||
                  (!text.trim() && attachments.length === 0 && pasteIssue === null) ||
                  runEngines.length === 0 ||
                  detecting
                }
              >
                {detecting
                  ? "检测中…"
                  : runEngines.length === 0
                    ? mode === "custom"
                      ? "请先勾选引擎"
                      : "无可用引擎"
                    : `开始检测（${runEngines.length} 引擎）`}
              </button>
              {detecting && (
                <button type="button" onClick={() => void cancelOrphanRun()}>
                  取消检测
                </button>
              )}
              <span className="row-spacer" />
              <button
                type="button"
                className="composer-import"
                onClick={pickAndImport}
                disabled={state !== "open" || attBusy || detecting}
                title="导入 docx / pdf / txt / md；也可以把文件直接拖进这个输入框"
              >
                <svg
                  viewBox="0 0 16 16"
                  width="14"
                  height="14"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M8.6 3.4 4.2 7.8a2.7 2.7 0 0 0 3.8 3.8l4.6-4.6a4 4 0 0 0-5.7-5.7L2.4 5.9" />
                </svg>
                {attBusy ? "解析中…" : "导入文件"}
              </button>
              <span className="composer-vram" title={residentTitle}>
                <span className={"vram-dot" + (catalog.resident.length > 0 ? " live" : "")} />
                驻留 {catalog.resident.length}
                <span className="vram-sep">·</span>
                {catalog.vram_used_mb > 0 ? (catalog.vram_measured ? "实测" : "估算") : "占用"}{" "}
                {fmtMb(catalog.vram_used_mb)}
                <span className="vram-sep">/</span>
                预算 {fmtMb(catalog.vram_budget_mb)}
                {catalog.vram_total_mb > 0 && (
                  <>
                    <span className="vram-sep">/</span>
                    显卡 {fmtMb(catalog.vram_total_mb)}
                  </>
                )}
              </span>
            </div>
          </div>
          <p className="hint">
            非「自定义」模式按输入语言自动选引擎（cpu 档自动避开慢引擎）；「自定义」在胶囊二级菜单里逐个勾选，勾选会记住。
            显存为 torch 实测占用，预算按「设置 → 常规 → 模型驻留」的档位换算。
          </p>

          {/* ---- orphaned run (round-19): reload killed the socket, the GPU
               job did not stop on its own. Name it, offer the stop. -------- */}
          {jobNotice && jobNotice.state === "running" && (
            <div className="warn job-strip">
              <span>
                上一轮检测仍在后台运行
                {jobNotice.engines.length > 0 &&
                  `（${jobNotice.engines.map(nameOf).join("、")}）`}
                {jobNotice.chars > 0 && `，${jobNotice.chars.toLocaleString("zh-CN")} 字`}
                {" · 已跑 "}
                {Math.max(1, Math.round(jobNotice.elapsed_s))}
                {" 秒 · 跑完会自动接回本窗口，也可以现在停掉"}
              </span>
              <button onClick={() => void cancelOrphanRun()}>取消这一轮</button>
            </div>
          )}

          {/* ---- results area: empty / streaming / done / failed states ---- */}
          {!detectResult && !detecting && (
            <div className="empty-state">
              <p>结果会显示在这里。</p>
              <p className="hint">
                上方粘贴文本、确认引擎后点「开始检测」。综合 AI 率置顶，正文按所选模型逐句上色；
                多个引擎时句子取最高分（宁可误报不漏报），悬停可看每个引擎的分数。
                高亮样式在「设置 → 外观与个性化」里调整。
              </p>
            </div>
          )}
          {detecting && Object.keys(partials).length === 0 && (
            <div className="empty-state">
              <p>推理中，首批句子马上出来…</p>
            </div>
          )}
          {(detectResult || Object.keys(partials).length > 0) && (
            <div className="results-panel">
              <div className="results-head">
                <span>检测结果</span>
              </div>
              <div className="results-body">
                {detectResult?.input && detectResult.input.attachments.length > 0 && (
                  <p className="input-source">
                    本次输入 {detectResult.input.chars.toLocaleString("zh-CN")} 字 · 含{" "}
                    {detectResult.input.attachments.join("、")}
                  </p>
                )}
                {warning && <p className="warn">⚠️ {warning}</p>}
                {detecting && <p className="hint">（流式接收中，以下数据还不是最终结果…）</p>}

                <CompositeCard
                  score={composite}
                  mode={hlMode}
                  results={okEngines}
                  spanCount={spans.length}
                  detailOpen={compDetail}
                  onToggleDetail={() => setCompDetail((v) => !v)}
                  nameOf={nameOf}
                />

                {modelSpans.length > 0 && (
                  <HighlightView text={resultText} spans={modelSpans} mode={hlMode} />
                )}
                {modelSpans.length === 0 && spans.length > 0 && (
                  <p className="hint">该模型没有句级结果——文档级引擎只给整篇分，不参与逐句上色。</p>
                )}

                {/* toolbar moved below the colour legend (round-28): these two
                    capsules filter/select the sentence list right underneath,
                    so they belong with 逐句明细, not above the document. */}
                <div className="result-controls">
                  <SelectCapsule
                    prefix="筛选："
                    value={sentFilter}
                    options={SENTENCE_FILTERS}
                    onChange={(id) => setSentFilter(id as SentenceFilter)}
                    title="逐句筛选"
                  />
                  <SelectCapsule
                    prefix="模型："
                    value={modelFilter}
                    options={modelOptions}
                    onChange={setModelFilter}
                    title="选择显示哪个模型的结果"
                    width={286}
                  />
                  <span className="result-count">
                    {sentenceRows.length} / {modelSpans.length} 句
                  </span>
                </div>

                {failedEngines.length > 0 && (
                  <div className="error-box">
                    <p>以下引擎未返回结果：</p>
                    <ul>
                      {failedEngines.map((r) => (
                        <li key={r.engine}>
                          {nameOf(r.engine)}: {r.status}
                          {r.detail ? ` — ${r.detail}` : ""}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {detectResult && okEngines.length === 0 && (
                  <div className="empty-state">
                    <p>所有引擎都没有成功返回——请检查引擎权重是否已下载、显存是否足够。</p>
                  </div>
                )}
                {detectResult && okEngines.length > 0 && (
                  <SentenceList
                    text={resultText}
                    rows={sentenceRows}
                    total={modelSpans.length}
                    mode={hlMode}
                    open={sentOpen}
                    onToggle={() => setSentOpen((v) => !v)}
                    note={sentenceNote}
                    nameOf={nameOf}
                  />
                )}
              </div>
            </div>
          )}
        </section>
      )}

      {shown === "work" && !catalog && !error && state !== "open" && (
        <div className="empty-state">
          <p>正在连接检测核心…</p>
          <p className="hint">
            桌面应用会自动完成连接。若长时间无响应，请尝试重启应用；
            端口 / Token 手动排查入口在「设置 → 开发者选项」。
          </p>
        </div>
      )}

      {/* ---- models view (round-6: split out of settings) ------------------ */}
      {shown === "models" && (
      <>
      <h2 className="view-title">模型管理</h2>

      <section className="settings-section" id="model-downloads">
        <h2>引擎下载</h2>
        <table>
          <thead>
            <tr>
              <th>选</th>
              <th>引擎</th>
              <th>状态</th>
              <th>体积</th>
              <th>风险</th>
              <th>默认</th>
              <th>进度</th>
            </tr>
          </thead>
          <tbody>
            {dlEngines.map(([eid, s]) => {
              const p = progress[eid];
              return (
                <tr key={eid}>
                  <td>
                    <input
                      type="checkbox"
                      checked={checked.has(eid)}
                      onChange={() => toggleCheck(eid, s.risk_level, s.risk_text_key)}
                      disabled={state !== "open" || dl?.running === true}
                    />
                  </td>
                  <td>
                    <div>{engineName(eid)}</div>
                    {s.repo && <div className="eng-repo">来源：{s.repo}</div>}
                  </td>
                  <td>{s.installed ? "已就位" : "未下载"}</td>
                  <td>{fmtBytes(s.total_bytes)}</td>
                  <td>{s.risk_level}</td>
                  <td>{s.default_selected ? "✓" : "—"}</td>
                  <td>{p ? `${fmtBytes(p.done)} / ${fmtBytes(p.total)}` : ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="row">
          <button onClick={startDownload} disabled={state !== "open" || checked.size === 0 || dl?.running === true}>
            下载选中（{checked.size}）
          </button>
          <button onClick={cancelDownload} disabled={dl?.running !== true}>
            取消
          </button>
          <span>{dl?.running ? `进行中: ${dl.active}` : "空闲"}</span>
          <span className="hint" id="dl-source-note">
            {dl?.running
              ? `下载源：${sourceHost(dl.source)}（本次任务）`
              : `下载源：${sourceHost(downloadSource)}（设置 → 常规 可改）`}
          </span>
        </div>
        <StoragePanel
          client={clientRef.current}
          engines={dlEngines}
          names={Object.fromEntries(catalog?.engines.map((e) => [e.id, e.name]) ?? [])}
          onDone={() => clientRef.current && void refresh(clientRef.current)}
          castProgress={castProgress}
          onError={setError}
          onNotice={(msg) => setJobLog((l) => [...l.slice(-20), msg])}
          jobRunning={dl?.running === true || detecting}
        />
        <pre className="log">{jobLog.join("\n") || "（任务日志）"}</pre>
      </section>

      {/* ---- custom model library (round-6): import + URL download ------- */}
      <section className="settings-section" id="custom-library">
        <h2>自定义模型库</h2>
        <p className="hint">
          两条通道：①导入本机已下好的模型目录——文件集能对上官方引擎时自动按
          manifest 就位（硬链接优先，跨盘自动转复制），对不上则登记进库备用；
          ②粘贴 URL 下载到 <code>models/custom/</code>（Range 断点续传 + sha256 记录）。
        </p>

        <div className="custom-grid">
          <div className="custom-card">
            <h3>本地导入</h3>
            <label className="field">
              模型目录路径
              <input
                value={importDir}
                onChange={(e) => setImportDir(e.target.value)}
                placeholder="例如 D:\hf-cache\openai-community--gpt2"
              />
            </label>
            <div className="row">
              <label>
                方式：
                <select value={importMode} onChange={(e) => setImportMode(e.target.value as "link" | "copy")}>
                  <option value="link">硬链接（同盘秒完成）</option>
                  <option value="copy">复制（保留原目录）</option>
                </select>
              </label>
              <label>
                名称（可选）：
                <input
                  value={importName}
                  onChange={(e) => setImportName(e.target.value)}
                  placeholder="默认取目录名"
                  style={{ width: 160 }}
                />
              </label>
            </div>
            <div className="row">
              <button
                className="primary"
                onClick={() => void runImport()}
                disabled={state !== "open" || !importDir.trim() || importBusy}
              >
                {importBusy ? "导入中…" : "导入"}
              </button>
              {importNotice && <span className="ok-note">{importNotice}</span>}
            </div>
          </div>

          <div className="custom-card">
            <h3>URL 下载</h3>
            <label className="field">
              下载地址
              <input
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="https://huggingface.co/<repo>/resolve/main/<file>"
              />
            </label>
            <div className="row">
              <label>
                名称（可选）：
                <input
                  value={urlName}
                  onChange={(e) => setUrlName(e.target.value)}
                  placeholder="默认取文件名"
                  style={{ width: 160 }}
                />
              </label>
            </div>
            <div className="row">
              <button
                className="primary"
                onClick={askUrlDownload}
                disabled={state !== "open" || !urlInput.trim() || customJob?.phase === "running" || customJob?.phase === "hashing"}
              >
                尝试下载
              </button>
            </div>
            {customJob && (
              <div className="custom-job">
                <span>
                  任务 {customJob.name} ·{" "}
                  {customJob.phase === "running"
                    ? "下载中"
                    : customJob.phase === "hashing"
                      ? "校验中（sha256）"
                      : customJob.phase === "done"
                        ? "完成"
                        : customJob.phase === "cancelled"
                          ? "已取消"
                          : "失败"}
                </span>
                {(customJob.phase === "running" || customJob.phase === "hashing") && (
                  <button onClick={() => void cancelCustomJob()}>取消</button>
                )}
                {customJob.total > 0 ? (
                  <span className="hint">
                    {" "}
                    {fmtBytes(customJob.done)} / {fmtBytes(customJob.total)}
                  </span>
                ) : (
                  customJob.done > 0 && (
                    /* mirrors sometimes answer without Content-Length (chunked),
                       so there is no percentage to show - report bytes only */
                    <span className="hint"> 已下载 {fmtBytes(customJob.done)}</span>
                  )
                )}
                {(customJob.resumed ?? 0) > 0 && (
                  <span className="hint custom-resumed">
                    断点续传自 {fmtBytes(customJob.resumed ?? 0)}
                  </span>
                )}
                {customJob.note && <p className="hint">{customJob.note}</p>}
              </div>
            )}
          </div>
        </div>

        {(customLib?.models.length ?? 0) > 0 && (
          <div className="custom-table-wrap">
          <table className="custom-table">
            <thead>
              <tr>
                <th>名称</th>
                <th>来源</th>
                <th>文件数</th>
                <th>体积</th>
                <th>说明</th>
                <th>时间</th>
              </tr>
            </thead>
            <tbody>
              {customLib!.models.map((m) => (
                <tr key={m.id}>
                  <td>
                    {m.engine ? engineName(m.engine) : m.name}
                    {m.engine && <span className="hint"> （官方引擎，已按 manifest 就位）</span>}
                  </td>
                  <td>
                    {m.source === "url"
                      ? `URL${m.mirrored ? " · 镜像" : " · 直连"}`
                      : "本地导入"}
                  </td>
                  <td>{m.files.length}</td>
                  <td>{fmtBytes(m.files.reduce((a, f) => a + (f.bytes || 0), 0))}</td>
                  <td className="hint custom-note">{m.note ?? ""}</td>
                  <td className="hint">{fmtImportedAt(m.imported_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        )}
        {customLib && customLib.models.length === 0 && (
          <p className="hint">库还是空的——导入一个目录或用 URL 下载开始。</p>
        )}
      </section>
      </>
      )}

      {/* ---- base runtime services (2026-11-27) ---------------------------- */}
      {shown === "runtime" && (
      <>
      <h2 className="view-title">基础服务组件</h2>
      <RuntimePanel
        status={runtimeStatus}
        dl={dl}
        downloadSource={downloadSource}
        onRefresh={refreshRuntime}
        onOpenModels={() => switchView("models")}
        onError={setError}
      />
      </>
      )}

      {/* ---- settings view (round-6: work/settings separation) ------------- */}
      {shown === "settings" && (
      <>
      <h2 className="view-title">设置</h2>

      {/* 常规 (2026-11-21): residency tier + sentence order habits ---------- */}
      <section className="settings-section" id="general">
        <h2>常规</h2>
        <div className="row wrap">
          <label>
            模型驻留：
            <select
              value={retention}
              onChange={(e) => void applyRetention(e.target.value as RetentionTier)}
            >
              {RETENTION_OPTIONS.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <button
            onClick={() => void releaseAllResident()}
            disabled={state !== "open" || (catalog?.resident.length ?? 0) === 0}
          >
            立即释放全部驻留
          </button>
          <label>
            逐句排序：
            <select
              value={sortPref}
              onChange={(e) => {
                const v = e.target.value as SortPref;
                setSortPref(v);
                try {
                  localStorage.setItem(SORT_KEY, v);
                } catch {
                  /* non-fatal */
                }
              }}
            >
              <option value="text">原文顺序（默认）</option>
              <option value="rate">按 AI 率降序</option>
            </select>
          </label>
          <label>
            下载源：
            <select
              value={downloadSource}
              onChange={(e) => {
                const v = e.target.value as DownloadSource;
                setDownloadSource(v);
                try {
                  localStorage.setItem(SOURCE_KEY, v);
                } catch {
                  /* non-fatal */
                }
              }}
            >
              {SOURCE_OPTIONS.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="hint">{RETENTION_OPTIONS.find((o) => o.id === retention)?.hint}</p>
        <p className="hint">
          下载源：{SOURCE_OPTIONS.find((o) => o.id === downloadSource)?.hint}
          影响「模型管理」下载和「基础服务组件」里的运行时下载入口；自定义模型库的
          URL 下载仍在开始前单独询问是否走镜像。
        </p>
        <p className="hint">
          「只看 AI 率较高 / 只看比较像人工」恒按比率排序；「全部句子」才跟随上面的排序偏好。
        </p>
        <div className="resident-box">
          <div className="resident-head">
            <span>当前驻留 · {catalog?.resident.length ?? 0} 个</span>
            <span className="hint">
              {(catalog?.vram_used_mb ?? 0) > 0 ? (catalog?.vram_measured ? "实测" : "估算") : "占用"}{" "}
              {fmtMb(catalog?.vram_used_mb ?? 0)} · 预算{" "}
              {fmtMb(catalog?.vram_budget_mb ?? 0)}
              {catalog?.vram_total_mb ? ` · 显卡 ${fmtMb(catalog.vram_total_mb)}` : ""}
            </span>
          </div>
          <div className="resident-chips">
            {(catalog?.resident_detail ?? []).map((r) => (
              <span
                className={"resident-chip" + (unloading === r.id ? " unloading" : "")}
                key={r.id}
                title={`${r.name} · 已空闲 ${Math.round(r.idle_s / 60)} 分钟`}
              >
                {r.name}
                <em>
                  {fmtMb(r.actual_mb ?? r.est_mb)}
                  {r.actual_mb === null ? " 估算" : ""}
                </em>
                {/* round-30: unload exactly this model; the × is inside the
                    capsule's right edge per the user sketch */}
                <button
                  type="button"
                  className="chip-x"
                  disabled={state !== "open" || unloading !== null}
                  aria-label={`从显存卸载 ${r.name}`}
                  title={`从显存卸载 ${r.name}`}
                  onClick={() => void unloadOne(r)}
                >
                  {unloading === r.id ? (
                    <span className="chip-spin" aria-hidden="true" />
                  ) : (
                    <svg viewBox="0 0 12 12" width="8" height="8" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
                      <path d="M3 3 9 9M9 3 3 9" />
                    </svg>
                  )}
                </button>
              </span>
            ))}
            {(catalog?.resident_detail ?? []).length === 0 && (
              <span className="hint">当前没有模型占用显存</span>
            )}
          </div>
        </div>
      </section>

      {/* appearance & personalization (round-6c): highlight + glass ring ---- */}
      <section className="settings-section" id="appearance">
        <h2>外观与个性化</h2>
        <div className="row">
          <label>
            高亮样式：
            <select value={hlMode} onChange={(e) => changeHlMode(e.target.value as HighlightMode)}>
              {HIGHLIGHT_MODES.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            液态玻璃：
            <select
              value={glassPref}
              onChange={(e) => changeGlassPref(e.target.value as GlassPreference)}
            >
              {GLASS_MODES.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="hint">
          高亮样式决定检测结果的六级色阶呈现；液态玻璃「自动」= 推理/下载时临时降为平面，
          cpu 档恒为平面。
        </p>
      </section>

      <DevOptions
        unlocked={devUnlocked}
        onUnlock={setDevUnlocked}
        conn={
          <div className="row conn-row">
            <label>
              端口 <input value={port} onChange={(e) => setPort(e.target.value)} style={{ width: 80 }} />
            </label>
            <label>
              Token <input value={token} onChange={(e) => setToken(e.target.value)} style={{ width: 160 }} />
            </label>
            {state === "idle" || state === "closed" ? (
              <button onClick={() => connect()}>连接</button>
            ) : (
              <button onClick={disconnect}>断开</button>
            )}
            <span>状态: {state}</span>
          </div>
        }
        env={
          <div className="env-rows">
            <div className="env-row"><span>Python</span><code>{health?.python_version ?? "—"}</code></div>
            <div className="env-row"><span>系统</span><code>{health?.platform ?? "—"}</code></div>
            <div className="env-row">
              <span>PyTorch</span>
              <code>
                {health?.torch_version ??
                  (health?.torch_installed === false ? "未安装" : "—")}
              </code>
            </div>
            <div className="env-row">
              <span>CUDA 构建</span>
              <code>{health?.cuda_build ?? "（CPU 版 wheel）"}</code>
            </div>
            <div className="env-row">
              <span>CUDA 可用</span>
              <code>{health?.cuda_available ? "是" : "否"}</code>
            </div>
            {health?.device_name && (
              <div className="env-row">
                <span>GPU</span>
                <code>{health.device_name}（{health.hw?.vram_total_mb ?? "?"} MB）</code>
              </div>
            )}
            <div className="env-row">
              <span>硬件档位</span>
              <code>
                {health?.hw?.tier ?? "—"}（source={health?.hw?.source ?? "—"} / wheel=
                {health?.hw?.wheel ?? "—"}）
              </code>
            </div>
            <div className="env-row">
              <span>内存 / 模型盘剩余</span>
              <code>
                {health?.hw?.ram_total_mb ?? "?"} MB / {health?.hw?.disk_free_mb ?? "?"} MB
              </code>
            </div>
            <div className="env-row">
              <span>SIDE_FORCE_TIER</span>
              <code>{health?.env?.force_tier ?? "（未设置）"}</code>
            </div>
            <div className="env-row">
              <span>SIDE_TOKEN</span>
              <code>{health?.env?.token_set ? "已设置（值不回显）" : "未设置"}</code>
            </div>
            <div className="env-row">
              <span>SIDE_EXTRA_ORIGINS</span>
              <code>{health?.env?.extra_origins ?? "（未设置）"}</code>
            </div>
            <div className="env-row">
              <span>模型目录</span>
              <code>{health?.env?.models_dir ?? "—"}</code>
            </div>
            <div className="env-row">
              <span>日志目录</span>
              <code>{health?.env?.logs_dir ?? "—"}</code>
            </div>
          </div>
        }
        logs={<LogExport getClient={() => clientRef.current} onNotice={pushNotice} />}
      />
      </>
      )}
        </div>

      {/* ---- format refused (round-14): the clipboard held an image/file ----- */}
      {formatOpen && (
        <div className="modal-backdrop" onClick={dismissFormat}>
          <div className="modal format-modal" onClick={(e) => e.stopPropagation()}>
            <h3>内容格式不支持</h3>
            <p>
              {pasteIssue === "image"
                ? "剪贴板里是一张图片，不是文字。"
                : pasteIssue === "file"
                  ? "剪贴板里是文件，不是文字。"
                  : "剪贴板里没有可用的文字内容。"}
            </p>
            <p className="hint">
              检测只接受纯文本，粘贴图片或文件不会把内容带进输入框。
              请复制文字后重新粘贴，或直接把文字输入到输入框。
            </p>
            <div className="modal-actions">
              <button className="primary" onClick={dismissFormat}>
                确认
              </button>
            </div>
          </div>
        </div>
      )}

      {/* D40: first import into a non-empty box asks once. Keeping the text is
          the default because the composed document is text-first. */}
      {importAsk && (
        <div className="modal-backdrop" onClick={() => setImportAsk(null)}>
          <div className="modal format-modal import-modal" onClick={(e) => e.stopPropagation()}>
            <h3>输入框里已有文字</h3>
            <p>
              输入框里已有 <b>{importAsk.chars.toLocaleString("zh-CN")}</b> 字。导入的文件会接在这些文字后面，
              拼成一篇一起检测。
            </p>
            <p className="hint">检测顺序：输入框文字 → 附件（按卡片从左到右）。</p>
            <div className="modal-actions">
              <button
                className="primary"
                onClick={() => {
                  const run = importAsk.run;
                  setImportAsk(null);
                  run(true);
                }}
              >
                保留文字并导入
              </button>
              <button
                onClick={() => {
                  const run = importAsk.run;
                  setImportAsk(null);
                  run(false);
                }}
              >
                清空文字后导入
              </button>
              <button onClick={() => setImportAsk(null)}>取消</button>
            </div>
          </div>
        </div>
      )}

      {/* ---- mirror prompt (custom URL download, two-step friendly) ------- */}
      {mirrorAsk && (
        <div className="modal-backdrop">
          <div className="modal mirror-modal">
            <h3>下载前的小提示</h3>
            <p>{mirrorAsk.hint}</p>
            <p className="hint">
              镜像站由第三方运营，内容与官方源一致但速度通常更快；直连则走官方源。
            </p>
            <div className="modal-actions">
              <button onClick={() => setMirrorAsk(null)}>取消</button>
              <button onClick={() => void runUrlDownload("direct")}>直连下载</button>
              <button
                className="primary"
                onClick={() => void runUrlDownload("mirror")}
                disabled={!mirrorAsk.hint.includes("hf-mirror.com")}
              >
                使用镜像站下载
              </button>
            </div>
          </div>
        </div>
      )}

      {gate && (
        <RiskDialog
          engineId={gate.engineId}
          riskLevel={gate.riskLevel}
          riskTextKey={gate.riskTextKey}
          onConfirmed={() => {
            setChecked((s) => new Set([...s, gate.engineId]));
            setGate(null);
          }}
          onCancelled={() => setGate(null)}
        />
      )}
      </main>

      {/* custom right-click menu + transient toast (2026-11-21) */}
      {ctxMenu && (
        <ContextMenu
          key={ctxMenu.seq}
          x={ctxMenu.x}
          y={ctxMenu.y}
          items={ctxMenu.items}
          onClose={() =>
            // identity-guarded: a slow exit animation from a previous menu
            // must not null out a newer menu opened in the meantime
            setCtxMenu((cur) => (cur?.seq === ctxMenu.seq ? null : cur))
          }
        />
      )}
      {toast && <div className="toast">{toast}</div>}
      {/* round-30: bottom-right stack (unload / export results, 15 s ttl) */}
      <Notices items={notices} onDismiss={dismissNotice} />
    </div>
  );
}
