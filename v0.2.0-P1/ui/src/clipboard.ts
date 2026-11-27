// Clipboard bridge (2026-11-21 right-click menu).
//
// round-14 user rule: pasting must never ask for permission. In the Tauri shell
// the native plugin is therefore the ONLY reader — `navigator.clipboard.readText()`
// makes WebView2 raise its "this page wants to read the clipboard" prompt, which
// is exactly the browser behaviour this desktop app must not have, so there is no
// fallback inside the shell at all. The web API is used only in a plain browser
// (Playwright verification). Writes keep the native command for the same reason.
import { invoke } from "@tauri-apps/api/core";

export type TextTarget = HTMLTextAreaElement | HTMLInputElement;

/** What a paste actually produced. `empty` = readable clipboard, no text in it. */
export type PasteOutcome = "ok" | "empty" | "failed";

const inTauri = (): boolean =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

const TEXT_INPUT_TYPES = new Set(["text", "search", "url", "email", "tel", "number", "password"]);

/** The focused-by-right-click text field, or null for read-only regions. */
export function textFieldOf(el: Element | null): TextTarget | null {
  const node = el?.closest("textarea, input");
  if (node instanceof HTMLTextAreaElement) return node;
  if (node instanceof HTMLInputElement) {
    return TEXT_INPUT_TYPES.has((node.type || "text").toLowerCase()) ? node : null;
  }
  return null;
}

/** True when the right-click landed on a non-empty selection. */
export function hasSelectionIn(el: TextTarget | null): boolean {
  if (el) {
    return el.selectionStart !== null && el.selectionEnd !== null && el.selectionStart !== el.selectionEnd;
  }
  return (window.getSelection()?.toString().length ?? 0) > 0;
}

export async function readClipboardText(): Promise<string | null> {
  // Native only inside the shell: no permission prompt, and no risky fallback.
  if (inTauri()) {
    try {
      const text = await invoke<string>("plugin:clipboard-manager|read_text");
      return typeof text === "string" ? text : "";
    } catch {
      // The plugin rejects when the clipboard holds no text (an image or a
      // file). That is "no text", not "unreadable" — and falling back to
      // `navigator.clipboard` here would raise WebView2's permission prompt
      // for exactly the case the user complained about.
      return "";
    }
  }
  try {
    const text = await navigator.clipboard.readText();
    if (typeof text === "string") return text;
  } catch {
    /* denied by the browser: nothing left to try */
  }
  return null;
}

export async function writeClipboardText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    /* fall through */
  }
  if (!inTauri()) return false;
  try {
    await invoke("plugin:clipboard-manager|write_text", { text });
    return true;
  } catch {
    return false;
  }
}

/** System clipboard -> caret. `empty` means the clipboard held no text at all
 *  (an image or a file), which the caller turns into a "格式不支持" explanation
 *  instead of silently pasting nothing. */
export async function pasteInto(el: TextTarget | null): Promise<PasteOutcome> {
  const text = await readClipboardText();
  if (text === null) return "failed";
  if (!el) return "failed";
  if (!text) return "empty";
  // a right-click on an unfocused box has no caret: append instead of
  // silently pasting at offset 0
  if (document.activeElement !== el) {
    el.focus();
    const end = el.value.length;
    el.setSelectionRange(end, end);
  }
  const start = el.selectionStart ?? el.value.length;
  const end = el.selectionEnd ?? start;
  // execCommand keeps the native undo stack; the setter fallback does not,
  // but it is the only path when the command is refused.
  if (document.execCommand("insertText", false, text)) return "ok";
  const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
  setter?.call(el, el.value.slice(0, start) + text + el.value.slice(end));
  el.dispatchEvent(new Event("input", { bubbles: true }));
  const pos = start + text.length;
  el.setSelectionRange(pos, pos);
  return "ok";
}

/** Selection -> clipboard. Native command first: it keeps the undo stack. */
export async function cutSelection(el: TextTarget | null): Promise<boolean> {
  if (!el) return false;
  el.focus();
  if (document.execCommand("cut")) return true;
  const text = el.value.slice(el.selectionStart ?? 0, el.selectionEnd ?? 0);
  if (!text) return false;
  const ok = await writeClipboardText(text);
  if (!ok) return false;
  const start = el.selectionStart ?? 0;
  const end = el.selectionEnd ?? start;
  el.setRangeText("", start, end, "end");
  el.dispatchEvent(new Event("input", { bubbles: true }));
  return true;
}

export async function copySelection(el: TextTarget | null): Promise<boolean> {
  if (el) el.focus();
  if (document.execCommand("copy")) return true;
  const text = el
    ? el.value.slice(el.selectionStart ?? 0, el.selectionEnd ?? 0)
    : (window.getSelection()?.toString() ?? "");
  if (!text) return false;
  return writeClipboardText(text);
}

export function selectAllIn(el: TextTarget | null): void {
  if (el) {
    el.focus();
    el.select();
    return;
  }
  const sel = window.getSelection();
  if (!sel) return;
  sel.removeAllRanges();
  const range = document.createRange();
  range.selectNodeContents(document.body);
  sel.addRange(range);
}
