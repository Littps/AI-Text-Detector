// D28⑤/D30 glass domain: three-tier degradation as pure functions.
//   glass  = liquid glass v2 recipe (light blur 4px + saturate 1.7, tint 0.42,
//            directional specular rim, inner volume, float shadows)
//   flat   = "replacement" surface: keep rim/radius, drop backdrop-filter,
//            raise bg opacity 0.42 -> 0.90 (300ms ease)
// Tiers: auto follows hardware (cpu tier stays flat) and busy state
// (detect/download); on/off override everything (D20 philosophy: gate,
// never lock).
export type GlassPreference = "auto" | "on" | "off";
export type GlassState = "glass" | "flat";

export const GLASS_MODES: { id: GlassPreference; label: string }[] = [
  { id: "auto", label: "自动（推理时降级）" },
  { id: "on", label: "始终开启" },
  { id: "off", label: "关闭" },
];

const LS_KEY = "aigc.glass.pref";

export function loadGlassPref(): GlassPreference {
  try {
    const v = localStorage.getItem(LS_KEY);
    if (v === "auto" || v === "on" || v === "off") return v;
  } catch {
    /* private mode etc. - default stays */
  }
  return "auto";
}

export function saveGlassPref(pref: GlassPreference): void {
  try {
    localStorage.setItem(LS_KEY, pref);
  } catch {
    /* non-fatal */
  }
}

/** Resolve the CSS state from user preference + hardware tier + busy flag. */
export function resolveGlass(
  pref: GlassPreference,
  tier: string | undefined,
  busy: boolean,
): GlassState {
  if (pref === "off") return "flat";
  if (pref === "on") return "glass";
  // auto: cpu tier stays flat permanently (D28⑤); busy flattens temporarily
  // (D30: CUDA inference competes with WebView compositing).
  return tier === "cpu" || busy ? "flat" : "glass";
}
