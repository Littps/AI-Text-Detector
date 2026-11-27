// Detect-mode profiles (round-6 user design): one capsule button next to the
// detect button opens an iOS-style popover with four modes. Non-custom modes
// resolve the engine set automatically from the pasted text's language and
// the hardware tier; custom keeps the manual engine chips.
//
// Language detection is a front-end CJK-ratio heuristic on purpose (NOT the
// r1 router): the router's protocol returns no language value, and a character
// census is free, deterministic, and accurate for zh/en discrimination.

export type DetectMode = "fast" | "balanced" | "precise" | "custom";

// Detection languages, in the order the capsule's tab strip shows them.
// round-32: this list is the single place to extend when another language
// gets detection support — add a row here, add its PROFILES column, and the
// capsule grows a tab (counts, empty state, "active" dot all follow).
export type DetectLang = "zh" | "en";

export const DETECT_LANGS: readonly { id: DetectLang; label: string }[] = [
  { id: "zh", label: "中文" },
  // self-named on purpose: language pickers read wrong when the target
  // language is itself translated
  { id: "en", label: "English" },
];

/** row tag wording for an engine's declared language */
export const ENGINE_LANG_TAG: Record<string, string> = {
  zh: "中文",
  en: "英文",
  multi: "多语",
};

export const MODE_LABELS: Record<DetectMode, string> = {
  fast: "高速模式",
  balanced: "均衡模式",
  precise: "高精度模式",
  custom: "自定义",
};

export const MODE_HINTS: Record<DetectMode, string> = {
  fast: "按输入语言自动选最快的引擎，秒级出结果",
  balanced: "按输入语言选高速 + 高性价比引擎组合",
  precise: "先确认文本语言，再按硬件选均衡与高精度引擎",
  custom: "手动勾选引擎（胶囊里逐个选）",
};

// mode -> language -> preferred engine ids (ordered best-first). Ids are
// filtered against the live catalogue at pick time, so not-yet-downloaded
// engines are silently skipped. fast-detectgpt / binoculars stay absent
// from AUTO profiles by design: they are exclusive, multi-GB method
// engines that must never be silently pulled into an automatic run -
// opting in via custom mode is the user's explicit decision.
const PROFILES: Record<Exclude<DetectMode, "custom">, Record<DetectLang, string[]>> = {
  fast: {
    zh: ["aigc-zhv3", "simpleai-zh"],
    en: ["modernbert-raid-mage", "vanguard-en"],
  },
  balanced: {
    // xiaoyu left balanced on 2026-11-25 (D44): it over-flags written
    // Chinese wholesale (human doc sentences 6/20 >= 0.5, max 0.9998) and
    // longer context made it worse, so it is a third opinion, not a default.
    zh: ["aigc-zhv3", "simpleai-zh"],
    en: ["modernbert-raid-mage", "desklib-en", "vanguard-en", "aigc-env3"],
  },
  precise: {
    zh: ["aigc-zhv3", "aigc-zhv3short", "simpleai-zh", "xiaoyu-zh"],
    // precise must be a SUPERSET of balanced, never a swap: the earlier list
    // dropped desklib-en and added gltr, so balanced/en and precise/en both
    // reported "4 engines" and high-precision looked like it changed nothing.
    // Fast classifiers first, then the document-level opinion, then the slow
    // method tail (it streams last on purpose).
    en: ["modernbert-raid-mage", "vanguard-en", "aigc-env3", "desklib-en", "gltr"],
  },
};

// Method-type engines carry a GPT-2 base forward pass: on cpu-tier hardware
// they are the slow tail, so they are dropped from every mode profile and
// the profile falls back to its fast set when nothing survives.
const CPU_TIER_SLOW = new Set(["gltr"]);

export function resolveModeEngines(
  mode: DetectMode,
  lang: DetectLang,
  tier: string | undefined,
  catalogueIds: Set<string>,
  usable?: Set<string>,
): string[] {
  if (mode === "custom") return [];
  let ids = PROFILES[mode][lang].filter((id) => catalogueIds.has(id));
  if (usable) ids = ids.filter((id) => usable.has(id));
  if (tier === "cpu") {
    // drop the slow method-type tail, then fall back to the fast set if
    // that leaves nothing (r1 fix: the old `|| catalogueIds.has(id)`
    // clause was always true and silently disabled this whole branch)
    const fast = PROFILES[lang === "zh" ? "balanced" : "fast"][lang].filter(
      (id) => catalogueIds.has(id) && (!usable || usable.has(id)),
    );
    ids = ids.filter((id) => !CPU_TIER_SLOW.has(id));
    if (ids.length === 0) ids = fast.filter((id) => !CPU_TIER_SLOW.has(id));
    if (ids.length === 0) ids = fast; // never return an empty cpu-tier set
  }
  return ids;
}

// Engines that can actually run right now: not the router (no AI segments)
// and — when download status is known — installed. No special-cased
// "unavailable" list since round-6e: every engine follows the same
// install-state rule (fdg's real implementation is wired backend-side).
// A missing dl payload means "don't filter on install state yet" (pre-first-
// refresh), matching the popover's historical lenient behaviour.
export function usableEngineIds(
  catalog: { engines: { id: string; kind: string }[] } | null,
  dl: { engines: Record<string, { installed: boolean }> } | null | undefined,
): Set<string> {
  const out = new Set<string>();
  for (const e of catalog?.engines ?? []) {
    if (e.kind === "router") continue;
    if (dl && dl.engines[e.id]?.installed !== true) continue;
    out.add(e.id);
  }
  return out;
}

/** display label for a detection language; DETECT_LANGS stays the only list */
export const detectLangLabel = (id: DetectLang): string =>
  DETECT_LANGS.find((l) => l.id === id)?.label ?? id;

/** 「中文文本」 vs 「English 文本」: a latin label needs the space, a CJK one
    must not get it — keeps the sentence correct for future languages too */
export function langWithNoun(id: DetectLang, noun: string): string {
  const label = detectLangLabel(id);
  return /[A-Za-z]$/.test(label) ? `${label} ${noun}` : `${label}${noun}`;
}

/** language census; null = the text carries no letters at all */
export function detectLangOrNull(text: string): DetectLang | null {
  let cjk = 0;
  let latin = 0;
  for (const ch of text) {
    const c = ch.codePointAt(0) ?? 0;
    if (c >= 0x4e00 && c <= 0x9fff) cjk++;
    else if ((c >= 0x41 && c <= 0x5a) || (c >= 0x61 && c <= 0x7a)) latin++;
  }
  // round-34: an empty box (or digits/punctuation only) has NO detection
  // language. Reporting "中文" there was a lie the capsule then had to
  // apologise for ("本次仍按中文检测") — the caller now renders 待判定 instead.
  if (cjk === 0 && latin === 0) return null;
  return cjk >= latin ? "zh" : "en";
}

// Small-print per-engine notes shown in the custom sub-sheet: what the model
// is good at, what it is NOT suited for. Honest, no marketing.
export const ENGINE_NOTES: Record<string, string> = {
  "simpleai-zh": "中文 · 快。通用中文首选；诗歌/歌词类短文本参考意义有限",
  "xiaoyu-zh": "中文 · 快。口语/网文域参考；书面中文系统性过判（人写文档探针 max 0.9998），只宜作第三意见",
  "aigc-zhv3": "中文 · 快。v3 新一代（对齐 DeepSeek-V3 / GPT-4）；正式文体偶发过判",
  "aigc-zhv3short": "中文 · 快。短句特化；极短人类句偶发过判（探针 0.60）",
  "modernbert-raid-mage": "英文 · 快。网文/社交媒体风格敏感",
  "desklib-en": "英文 · 快。文档级引擎——全篇一个分数，不做逐句高亮",
  "vanguard-en": "英文 · 快。正式文体（新闻/论文）敏感",
  "aigc-env3": "英文 · 快。v3 新一代；口语/邮件体偶发过判（探针 0.59）",
  gltr: "英文 · 慢。方法型（GPT-2 统计特征），学术腔敏感；对维基式正式文体误报偏高；显存开销较大",
  "fast-detectgpt": "英文 · 重（独占显存）。方法型（条件概率曲率）；需先在模型管理下载 10.67GB 基座，分数未经重验标定",
  binoculars: "英文 · 极重（独占显存 28GB）。方法型（双模型交叉困惑度）；实现已落地，4bit 量化未实现，8GB 卡会 OOM",
  "r1-lang-router": "语言路由基础设施——不产出 AI 率，不参与检测",
};
