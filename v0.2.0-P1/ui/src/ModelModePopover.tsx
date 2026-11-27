// iOS-style mode popover (round-6): anchored to the capsule trigger with
// viewport-space positioning and real flip-probing — below the anchor when
// there is room, above when there is not, re-probed on every sub-sheet
// switch. Spring-eased scale+fade entrance; window scroll/resize closes it
// (mobile-convention), which also keeps the fixed anchor honest.
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type RefObject } from "react";
import type { EnginesPayload, DlStatusPayload, CustomModelEntry } from "./types";
import { fmtBytes } from "./ws";
import { SPRING_CLOSE } from "./spring";
import {
  ENGINE_NOTES,
  MODE_HINTS,
  MODE_LABELS,
  resolveModeEngines,
  usableEngineIds,
  type DetectMode,
} from "./modelModes";

// friendly display name helper (round-6: no raw ids in the mode UI)
const nameOf = (id: string, named: { id: string; name: string }[]): string =>
  named.find((e) => e.id === id)?.name || id;

// row language tag: the engine names already carry (EN)/(CN), but the picker
// states the detection text type explicitly so a row reads without decoding
const LANG_LABEL: Record<string, string> = { zh: "中文", en: "英文", multi: "多语" };

// ENGINE_NOTES already open with "中文 · 快。" — the tag carries that part now,
// so the row keeps only the applicability sentence (no duplicated language)
const noteOf = (id: string): string =>
  (ENGINE_NOTES[id] ?? "").replace(/^(中文|英文|多语)\s*·\s*/, "");

interface Props {
  mode: DetectMode;
  preciseLang: "zh" | "en";
  catalog: EnginesPayload;
  tier: string | undefined;
  dl: DlStatusPayload | null;
  picks: Set<string>;
  /** custom-library registry entries: shown read-only, they are not engines yet */
  customModels: CustomModelEntry[];
  // stable ref to the capsule trigger; the popover reads its rect itself
  anchorRef: RefObject<HTMLButtonElement | null>;
  onPickToggle: (id: string) => void;
  /** bulk replace for select-all / clear / invert (custom sheet) */
  onReplacePicks: (ids: string[]) => void;
  onCommit: (mode: DetectMode, preciseLang: "zh" | "en") => void;
  onClose: () => void;
}

export function ModelModePopover({
  mode,
  preciseLang,
  catalog,
  tier,
  dl,
  picks,
  customModels,
  anchorRef,
  onPickToggle,
  onReplacePicks,
  onCommit,
  onClose,
}: Props) {
  // already in custom mode? the capsule is the "edit engines" affordance, so
  // open straight into the pick sheet instead of making the user re-pick
  // 自定义 from the mode list every time (‹ 返回 still reaches the list).
  const [sub, setSub] = useState<null | "precise" | "custom">(() =>
    mode === "custom" ? "custom" : null,
  );
  const popRef = useRef<HTMLDivElement>(null);
  const scrimRef = useRef<HTMLDivElement>(null);
  // dismissal runs as a spring first, then unmounts: closing the sheet on a
  // hard cut (the old behaviour) read as a paint glitch rather than a motion
  const closingRef = useRef(false);
  const [closing, setClosing] = useState(false);
  // initial guess: below-left; corrected before first paint by the probe
  const [place, setPlace] = useState<{ left: number; dir: "above" | "below"; maxH: number }>(
    () => ({ left: anchorRef.current?.getBoundingClientRect().left ?? 8, dir: "below", maxH: 420 }),
  );

  // flip-probe: measure the rendered popover and pick the side with room,
  // re-run when the sub-sheet changes height (precise/custom sheets are tall).
  // When neither side fits (short viewports) the larger side wins and the
  // popover clamps to that space, scrolling internally.
  useLayoutEffect(() => {
    const el = popRef.current;
    const cap = anchorRef.current?.getBoundingClientRect();
    if (!el || !cap) return;
    const h = el.offsetHeight;
    const w = el.offsetWidth;
    const spaceBelow = window.innerHeight - cap.bottom;
    const spaceAbove = cap.top;
    // the custom sheet is the tall one: give it the roomier side instead of
    // the "prefer below" rule, otherwise a bottom-anchored capsule squeezes it
    const dir: "above" | "below" =
      sub === "custom"
        ? spaceAbove >= spaceBelow
          ? "above"
          : "below"
        : spaceBelow >= h + 14 || spaceBelow >= spaceAbove
          ? "below"
          : "above";
    const left = Math.max(8, Math.min(cap.left, window.innerWidth - w - 8));
    // never floor above the room that actually exists: a hard minimum used to
    // push the sheet past the top edge on short viewports (top went negative)
    const avail = Math.max(120, (dir === "below" ? spaceBelow : spaceAbove) - 14);
    const maxH = Math.min(window.innerHeight - 16, avail);
    setPlace((p) =>
      p.dir === dir && Math.abs(p.left - left) < 1 && Math.abs(p.maxH - maxH) < 2
        ? p
        : { left, dir, maxH },
    );
  }, [sub, anchorRef]);

  // Spring dismissal: play the exit curve on the popover + scrim, then hand
  // control back to the parent (which unmounts us). Guarded so a click on the
  // scrim, a scroll event and a mode commit cannot all race into onClose().
  const requestClose = useCallback(() => {
    if (closingRef.current) return;
    closingRef.current = true;
    const el = popRef.current;
    const scrim = scrimRef.current;
    if (!el || typeof el.animate !== "function") {
      onClose();
      return;
    }
    setClosing(true);
    scrim?.animate([{ opacity: 1 }, { opacity: 0 }], {
      duration: Math.min(SPRING_CLOSE.duration, 220),
      easing: "ease-out",
      fill: "forwards",
    });
    const anim = el.animate(
      [
        { opacity: 1, transform: "scale(1) translateY(0)" },
        { opacity: 0, transform: "scale(0.955) translateY(4px)" },
      ],
      { duration: SPRING_CLOSE.duration, easing: SPRING_CLOSE.easing, fill: "forwards" },
    );
    anim.onfinish = () => onClose();
  }, [onClose]);

  // fixed positioning goes stale only when the window/document itself scrolls
  // -> close. Capture mode would ALSO see scrolling inside the popover (the
  // scroll event does not bubble, but capture observes every target), closing
  // the sheet mid-read whenever the user scrolls a tall sub-sheet - a real
  // bug hit while automating the custom sub-sheet (round-6d).
  useEffect(() => {
    window.addEventListener("scroll", requestClose);
    window.addEventListener("resize", requestClose);
    return () => {
      window.removeEventListener("scroll", requestClose);
      window.removeEventListener("resize", requestClose);
    };
  }, [requestClose]);

  const catalogueIds = new Set(catalog.engines.map((e) => e.id));
  // what the profiles may actually use right now (installed, non-router):
  // the precise sheet must show the same set the run will execute, otherwise
  // a fixed four-name list looks "locked" even when something is missing
  const usable = usableEngineIds(catalog, dl);
  const selectableIds = catalog.engines
    .filter((e) => e.kind !== "router" && (!dl || dl.engines[e.id]?.installed === true))
    .map((e) => e.id);
  const selectedCount = selectableIds.filter((id) => picks.has(id)).length;
  const rows: DetectMode[] = ["fast", "balanced", "precise", "custom"];

  const posStyle: CSSProperties =
    place.dir === "below"
      ? {
          left: place.left,
          top: (anchorRef.current?.getBoundingClientRect().bottom ?? 0) + 10,
          maxHeight: place.maxH,
        }
      : {
          left: place.left,
          bottom:
            window.innerHeight - (anchorRef.current?.getBoundingClientRect().top ?? 0) + 10,
          maxHeight: place.maxH,
        };

  // precise rows must advertise the real count (and the real names), not the
  // profile's nominal list: that mismatch is what made "4 engines" look stuck
  const preciseHint = (lang: "zh" | "en"): string => {
    const ids = resolveModeEngines("precise", lang, tier, catalogueIds, usable);
    if (ids.length === 0) return "无可用引擎";
    return `${ids.length} 引擎 · ${ids.map((id) => nameOf(id, catalog.engines)).join(", ")}`;
  };

  return (
    <>
      <div ref={scrimRef} className="popover-scrim" onClick={requestClose} />
      <div
        ref={popRef}
        className={
          "mode-popover " + place.dir + (closing ? " closing" : "") + (sub === "custom" ? " custom-sheet" : "")
        }
        style={{ ...posStyle, transformOrigin: place.dir === "below" ? "top left" : "bottom left" }}
        role="menu"
      >
        {sub === null && (
          <div className="mode-list">
            {rows.map((m, i) => (
              <button
                key={m}
                className={"mode-row" + (mode === m ? " active" : "")}
                style={{ animationDelay: `${i * 30}ms` }}
                onClick={() => {
                  if (m === "precise") setSub("precise");
                  else if (m === "custom") setSub("custom");
                  else {
                    onCommit(m, preciseLang);
                    requestClose();
                  }
                }}
              >
                <span className="mode-row-name">{MODE_LABELS[m]}</span>
                <span className="mode-row-hint">{MODE_HINTS[m]}</span>
                {mode === m && <span className="mode-row-check">✓</span>}
              </button>
            ))}
            <p className="mode-foot hint">
              非「自定义」模式按输入语言自动选引擎（cpu 档会避开慢方法型引擎）。
            </p>
          </div>
        )}

        {sub === "precise" && (
          <div className="mode-sub">
            <button className="mode-back" onClick={() => setSub(null)}>
              ‹ 返回
            </button>
            <p className="mode-sub-title">确认要检测的文本语言</p>
            <button
              className={"mode-row" + (mode === "precise" && preciseLang === "zh" ? " active" : "")}
              onClick={() => {
                onCommit("precise", "zh");
                requestClose();
              }}
            >
              <span className="mode-row-name">中文文本</span>
              <span className="mode-row-hint">
                {preciseHint("zh")}
              </span>
            </button>
            <button
              className={"mode-row" + (mode === "precise" && preciseLang === "en" ? " active" : "")}
              onClick={() => {
                onCommit("precise", "en");
                requestClose();
              }}
            >
              <span className="mode-row-name">英文文本</span>
              <span className="mode-row-hint">
                {preciseHint("en")}
              </span>
            </button>
          </div>
        )}

        {sub === "custom" && (
          <div className="mode-sub mode-custom">
            <div className="mode-custom-head">
              <div className="mode-custom-nav">
                <button className="mode-back" onClick={() => setSub(null)}>
                  ‹ 返回
                </button>
                <span className="mode-custom-title">手动勾选引擎</span>
              </div>
              <div className="mode-bulk">
                <span className="mode-bulk-count">
                  已选 {selectedCount} / {selectableIds.length}
                </span>
                <button
                  type="button"
                  disabled={selectableIds.length === 0 || selectedCount === selectableIds.length}
                  onClick={() => onReplacePicks(selectableIds)}
                >
                  全选
                </button>
                <button
                  type="button"
                  disabled={selectedCount === 0}
                  onClick={() => onReplacePicks([])}
                >
                  清空
                </button>
                <button
                  type="button"
                  disabled={selectableIds.length === 0}
                  onClick={() => onReplacePicks(selectableIds.filter((id) => !picks.has(id)))}
                >
                  反选
                </button>
              </div>
            </div>
            {/* rows: checkbox / name + text-type tag / applicability note below.
                Kept as one scroll area so the head and the done button stay put. */}
            <div className="mode-pick-list">
              {catalog.engines.map((e, i) => {
                const router = e.kind === "router";
                const notInstalled = dl ? dl.engines[e.id]?.installed !== true : false;
                const disabled = router || notInstalled;
                const active = picks.has(e.id);
                return (
                  <label
                    key={e.id}
                    className={
                      "mode-pick" + (active ? " active" : "") + (disabled ? " disabled" : "")
                    }
                    style={{ animationDelay: `${Math.min(i, 8) * 22}ms` }}
                  >
                    <input
                      type="checkbox"
                      disabled={disabled}
                      checked={active}
                      onChange={() => onPickToggle(e.id)}
                    />
                    <span className="pick-box" aria-hidden="true">
                      <svg viewBox="0 0 12 12" width="10" height="10">
                        <path
                          d="M1.7 6.3 4.4 9l5.9-6.2"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.9"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </span>
                    <span className="pick-body">
                      <span className="pick-head">
                        <span className="pick-name" title={e.id}>
                          {e.name || e.id}
                        </span>
                        <span className="pick-tag">{LANG_LABEL[e.lang] ?? e.lang}</span>
                        {e.loaded && <span className="pick-tag pick-tag-live">驻留</span>}
                        {notInstalled && !router && (
                          <span className="pick-tag pick-tag-off">未下载</span>
                        )}
                      </span>
                      <span className="pick-note" title={noteOf(e.id)}>
                        {router ? "语言路由基础设施，不产出 AI 率，恒不参与检测" : noteOf(e.id)}
                      </span>
                    </span>
                  </label>
                );
              })}
              {customModels.length > 0 && (
                <div className="mode-lib">
                  <p className="mode-sub-title">
                    自定义库（{customModels.length}）· 已登记，未接入推理
                  </p>
                  {customModels.map((m) => (
                    <div className="mode-lib-row" key={m.id} title={m.dir}>
                      <span className="mode-row-name">{m.name}</span>
                      <span className="mode-row-hint">
                        {m.files.length} 个文件 ·{" "}
                        {fmtBytes(m.files.reduce((a, f) => a + (f.bytes || 0), 0))}
                        {m.engine ? ` · 已就位官方引擎 ${m.engine}` : " · 未匹配官方引擎"}
                      </span>
                    </div>
                  ))}
                  <p className="mode-foot hint">
                    自定义库条目只是文件集登记，不参与检测；接入推理实现后才能勾选。
                    管理入口在「模型管理 → 自定义模型库」。
                  </p>
                </div>
              )}
            </div>
            <div className="mode-custom-foot">
              <button
                className="mode-done"
                onClick={() => {
                  onCommit("custom", preciseLang);
                  requestClose();
                }}
              >
                完成（{selectedCount} 引擎）
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
