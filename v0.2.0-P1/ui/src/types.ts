// Protocol v1 types - mirrors backend/protocol.py + server.py payloads.
// Every WS message: {v, id, type, ts, payload}. Responses carry type
// "res" with payload or {error}; server-pushed events use type "event"
// with fixed id "dl-event".

export interface Envelope<T = unknown> {
  v: number;
  id: string;
  type: string;
  ts: number;
  payload: T;
}

export interface HwBlock {
  cuda: boolean;
  capability: number[] | null;
  gpu_name: string | null;
  vram_total_mb: number;
  ram_total_mb: number;
  /** CPU model string from the registry / /proc/cpuinfo (round-14) */
  cpu_name?: string | null;
  cpu_threads?: number;
  disk_free_mb: number;
  tier: string;
  wheel: string;
  source: string;
  driver_version?: string | null;
  note?: string;
}

export interface HealthPayload {
  status: string;
  python_version: string;
  platform: string;
  hw: HwBlock;
  torch_installed: boolean;
  torch_version?: string;
  cuda_build?: string | null;
  cuda_available?: boolean;
  device_name?: string;
  env?: {
    force_tier: string | null;
    token_set: boolean;
    extra_origins: string | null;
    models_dir: string;
    /** round-30: where the daily sidecar logs live (dev options) */
    logs_dir?: string;
  };
}

export interface EngineInfo {
  id: string;
  name: string;
  lang: string;
  kind: string;
  est_vram_mb: number;
  /** real footprint captured around load() (null = not measured) */
  actual_vram_mb: number | null;
  exclusive: boolean;
  verified: boolean;
  loaded: boolean;
  resident: boolean;
}

/** "how long a model stays in VRAM after a detect" tiers (backend lru.py) */
export type RetentionTier = "forever" | "default" | "30m" | "10m" | "5m" | "none";

export const RETENTION_OPTIONS: { id: RetentionTier; label: string; hint: string }[] = [
  { id: "forever", label: "常驻（永久）", hint: "用到显卡几乎满载为止，不为新模型主动腾地方" },
  { id: "default", label: "常驻（默认）", hint: "装不下时先换出「语言不对口」的旧模型，再按最久未用" },
  { id: "30m", label: "驻留 30 分钟", hint: "最后一次使用后 30 分钟释放显存" },
  { id: "10m", label: "驻留 10 分钟", hint: "最后一次使用后 10 分钟释放显存" },
  { id: "5m", label: "驻留 5 分钟", hint: "最后一次使用后 5 分钟释放显存" },
  { id: "none", label: "不驻留", hint: "每次检测结束立即释放，下次检测重新加载" },
];

export interface ResidentEntry {
  id: string;
  name: string;
  lang: string;
  est_mb: number;
  actual_mb: number | null;
  idle_s: number;
}

/** Measured residency block: engines / detect / resident events share it. */
export interface VramBlock {
  resident: string[];
  resident_detail: ResidentEntry[];
  vram_used_mb: number;
  vram_budget_mb: number;
  vram_total_mb: number;
  vram_measured: boolean;
  retention: RetentionTier;
  retention_ttl_s: number | null;
}

export interface EnginesPayload extends VramBlock {
  engines: EngineInfo[];
}

// ---- round-30: per-model unload + log export ----------------------------

/** answer of ``resident/unload {engine_id}`` (a superset of VramBlock) */
export interface UnloadPayload extends VramBlock {
  unloaded: string[];
  results: { id?: string; ok?: boolean; detail?: string }[];
}

/** the three ranges the export dialog offers (backend validates the same set) */
export type LogRange = 1 | 3 | 7;

export interface LogExportPayload {
  text: string;
  days: LogRange;
  from: string;
  to: string;
  files: number;
  lines: number;
  bytes: number;
  truncated: boolean;
}

// ---- D25 storage domain -------------------------------------------------

/**
 * D36 download source preference (settings -> 常规). It only reorders the
 * host chain: the other host stays as the automatic fallback.
 */
export type DownloadSource = "official" | "mirror";

export type StorageMode = "keep-original" | "dual" | "fp16-only";
export const STORAGE_MODES: StorageMode[] = ["keep-original", "dual", "fp16-only"];

export interface StorageStatus {
  storage_mode: StorageMode;
  fp32_present: boolean;
  fp16_present: boolean;
  installed: boolean;
  recast_needed: boolean;
  stale_cast: boolean;
  dtype_options: ("float16" | "float32")[];
}

export interface DlEngineStatus {
  repo?: string;
  installed: boolean;
  total_bytes: number;
  default_selected: boolean;
  risk_level: "none" | "medium" | "high";
  risk_text_key: string | null;
  min_vram_mb: number;
  storage: StorageStatus;
}

export interface DlStatusPayload {
  engines: Record<string, DlEngineStatus>;
  active: string | null;
  running: boolean;
  /** last source preference the backend was asked to use (D36) */
  source?: DownloadSource;
}

export interface DownloadStartPayload {
  started: string[];
  skipped: { engine_id: string; reason: string }[];
}

export interface DownloadCancelPayload {
  cancel_requested: boolean;
}

export interface SetModePayload {
  engine_id: string;
  storage: StorageStatus;
}

export interface CastStartPayload {
  cast_started: string[];
  mode: StorageMode;
}

export interface CleanupPayload {
  engine_id: string;
  removed: string[];
}

export interface StorageUninstallPayload {
  engine_id: string;
  removed: string[];
  bytes_freed: number;
  leftovers: string[];
  shared_kept: string[];
}

// ---- runtime component audit --------------------------------------------

export type RuntimeComponentState = "ready" | "warning" | "missing";

export interface RuntimeComponent {
  id: "python" | "websockets" | "torch" | "transformers" | "vc-runtime" | string;
  label: string;
  role: string;
  state: RuntimeComponentState;
  version: string | null;
  detail: string;
  location: string | null;
  managed_by: string;
  required: boolean;
}

export interface RuntimeDll {
  name: string;
  found: boolean;
  path: string | null;
  bytes: number;
}

export interface RuntimeStatusPayload {
  generated_at: string;
  mode: "bundled" | "development" | "external" | string;
  python_version: string;
  python_executable: string;
  app_root: string;
  runtime_root: string;
  site_packages: string;
  model_root: string;
  recommended_wheel: "cu129" | "cpu" | string;
  /** Active PyTorch wheel family: CPU-only, CUDA build, or unavailable. */
  torch_variant: "cpu" | "cuda" | "unavailable";
  hardware: {
    cuda: boolean;
    gpu_name: string | null;
    driver_version: string | null;
    capability: number[] | null;
    tier: string | null;
    wheel: string;
    source: string | null;
  };
  local_service_ready: boolean;
  inference_ready: boolean;
  components: RuntimeComponent[];
  dlls: RuntimeDll[];
  notes: string[];
}

// ---- detect (D29: full-sentence return + doc_score + streaming) ---------

export interface DetectSegment {
  offset: number;
  length: number;
  score: number;
}

export interface DetectResult {
  engine: string;
  status: "ok" | "weights_missing" | "error" | "unknown_engine";
  segments: DetectSegment[];
  doc_score: number | null;
  detail?: string;
}

export interface DetectPayload {
  /** Job that produced this result; used to reject a stale parked reply. */
  job_id?: number;
  results: DetectResult[];
  /** What the detector actually read (D40): composed chars + attachment names.
   *  `text` only comes back when attachments were composed — the UI already owns
   *  pasted text, but file bodies live in the sidecar and the sentence list
   *  needs them to render. */
  input?: { chars: number; attachments: string[]; text?: string };
  vram_used_mb: number;
  vram_budget_mb: number;
}

// ---- detect lifecycle (round-19) ---------------------------------------

/** Public state of the sidecar's single detect slot. Never carries text. */
export interface DetectJobInfo {
  state: "idle" | "running" | "done" | "error" | "cancelled";
  job_id: number;
  engines: string[];
  current: string | null;
  chars: number;
  started_at: number | null;
  finished_at: number | null;
  elapsed_s: number;
  cancel_requested: boolean;
  reason: string | null;
  detail: string | null;
  /** Round-20: a finished reply is parked because the socket that asked for
   *  it died mid-run. The page that took over fetches it with detect/result. */
  has_result: boolean;
  /** Job id attached to the parked reply, when one exists. */
  result_job_id?: number | null;
}

// ---- file import (D40) --------------------------------------------------

export interface AttachmentMeta {
  id: string;
  name: string;
  ext: string;
  chars: number;
  bytes: number;
  notes: string[];
}

export interface FileImportPayload {
  items: AttachmentMeta[];
  errors: { name: string; error: string }[];
  all: AttachmentMeta[];
}
export type DlEvent =
  | { kind: "download_engine_start"; engine_id: string; files: number; source?: DownloadSource }
  | {
      kind: "download_progress";
      engine_id: string;
      dest: string;
      done_bytes: number;
      total_bytes: number;
    }
  | { kind: "download_file_done"; engine_id: string; dest: string }
  | { kind: "download_engine_done"; engine_id: string }
  | { kind: "download_cancelled"; engine_id: string }
  | { kind: "download_error"; engine_id: string; dest: string; error: string }
  | { kind: "cast_start"; engine_id: string; mode: StorageMode }
  | {
      kind: "cast_progress";
      engine_id: string;
      dest: string;
      done_bytes: number;
      total_bytes: number;
    }
  | { kind: "cast_done"; engine_id: string; mode: StorageMode }
  | { kind: "cast_cancelled"; engine_id: string }
  | { kind: "cast_error"; engine_id: string; error: string }
  | { kind: "detect_part"; engine_id: string; segments: DetectSegment[] }
  // residency changed server-side (idle sweep / manual unload): re-read engines
  | ({ kind: "resident" } & VramBlock)
  // custom library job events (round-6): phase + job snapshot in one frame
  | {
      kind: "custom-dl";
      phase: "start" | "progress" | "hashing" | "done" | "error" | "cancelled";
      name: string;
      url: string;
      file?: string;
      done: number;
      total: number;
      /** bytes carried over from a cancelled attempt's .part */
      resumed?: number;
      note?: string | null;
    };

// D20 risk levels that gate the double-dialog confirm (>= medium).
export const RISK_DUAL_DIALOG: ReadonlySet<string> = new Set(["medium", "high"]);

// ---- custom model library (round-6: local import + URL download) --------

export interface CustomFileEntry {
  name: string;
  bytes: number;
  sha256?: string | null;
  via?: string;
}

export interface CustomModelEntry {
  id: string;
  name: string;
  source: "local" | "url";
  origin_url: string | null;
  mirrored: boolean;
  note?: string | null;
  dir: string;
  engine?: string;
  files: CustomFileEntry[];
  imported_at: string;
}

export interface CustomJob {
  phase: "running" | "hashing" | "done" | "error" | "cancelled";
  name: string;
  url: string;
  file?: string;
  done: number;
  total: number;
  /** bytes inherited from a cancelled attempt's .part (0 when starting fresh) */
  resumed?: number;
  note?: string | null;
}

export interface CustomListPayload {
  models: CustomModelEntry[];
  job: CustomJob | null;
}

export interface CustomImportResult {
  kind: "engine" | "custom";
  engine?: string;
  name?: string;
  note: string;
  transferred: { name: string; dest?: string; via: string }[];
  verify?: {
    installed?: boolean;
    recast_needed?: boolean;
    stale_cast?: boolean;
    mode?: string;
  };
}
