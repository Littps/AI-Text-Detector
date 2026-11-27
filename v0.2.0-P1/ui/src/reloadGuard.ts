/**
 * Round-19 hardening #1: the packaged window must not reload itself.
 *
 * A page reload while a detect runs kills the only socket that owns the reply
 * (the sidecar keeps burning the GPU until the cancel watcher fires), and the
 * one time the WebView2 compositor went white it happened right after two
 * reloads in a row. Users have no reason to reload a shipped app: the escape
 * hatch for a stuck UI is the tray menu (Rust side), which reloads on purpose.
 *
 * Kept as a plain function so the verification harness can install it on a
 * normal browser page and assert `defaultPrevented` without a Tauri shell.
 */

type KeyLike = Pick<KeyboardEvent, "key" | "ctrlKey" | "metaKey" | "altKey">;

/** F5 (any modifier), Ctrl+R / Cmd+R, Ctrl+Shift+R. Alt combos are left alone. */
export function isReloadKey(e: KeyLike): boolean {
  if (e.altKey) return false;
  if (e.key === "F5") return true;
  return (e.ctrlKey || e.metaKey) && (e.key === "r" || e.key === "R");
}

/** Swallow the reload shortcuts; returns an uninstall handle (tests/HMR). */
export function installReloadGuard(target: Window = window): () => void {
  const onKeyDown = (e: KeyboardEvent) => {
    if (!isReloadKey(e)) return;
    e.preventDefault();
    e.stopPropagation();
  };
  target.addEventListener("keydown", onKeyDown, true);
  return () => target.removeEventListener("keydown", onKeyDown, true);
}
