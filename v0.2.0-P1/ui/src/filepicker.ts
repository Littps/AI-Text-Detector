// File import bridge (D40).
//
// Two routes, one protocol. Inside the Tauri shell the native dialog returns
// absolute paths and the sidecar opens them itself — the only shape that keeps
// the document on disk and lets a 300-page PDF be parsed without copying it
// through the webview. In a plain browser (how the UI is verified, and how a
// future web build would run) a webview cannot expose a real path, so the bytes
// are read here and posted as base64 instead.
import { open } from "@tauri-apps/plugin-dialog";

/** Extensions the sidecar can parse. Keep in sync with backend/filereader.py. */
export const ACCEPTED_EXTS = [".docx", ".pdf", ".txt", ".md"];
export const ACCEPT_ATTR = ACCEPTED_EXTS.join(",");

export const inTauri = (): boolean =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export interface FileBlob {
  name: string;
  data: string; // base64
}

/** Native multi-select. Returns [] when the user cancels or outside the shell. */
export async function pickPaths(): Promise<string[]> {
  if (!inTauri()) return [];
  const picked = await open({
    multiple: true,
    directory: false,
    title: "选择要检测的文档",
    filters: [{ name: "文档", extensions: ["docx", "pdf", "txt", "md"] }],
  });
  if (!picked) return [];
  return Array.isArray(picked) ? picked : [picked];
}

/** Hidden <input type=file> for the browser path. */
export function pickBrowserFiles(): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ACCEPT_ATTR;
    input.style.display = "none";
    document.body.appendChild(input);

    let settled = false;
    const finish = (files: File[]) => {
      if (settled) return;
      settled = true;
      input.remove();
      resolve(files);
    };
    input.addEventListener("change", () => finish(Array.from(input.files ?? [])));
    // Chrome/Edge fire "cancel" when the picker is dismissed; older engines do
    // not, and the promise then resolves on the next change — acceptable here
    // because nothing else is blocked while the picker is open.
    input.addEventListener("cancel", () => finish([]));
    input.click();
  });
}

export async function toBlobs(files: File[]): Promise<FileBlob[]> {
  const out: FileBlob[] = [];
  for (const f of files) {
    out.push({ name: f.name, data: base64(new Uint8Array(await f.arrayBuffer())) });
  }
  return out;
}

/** 20 MiB per file, mirroring backend/filereader.MAX_BYTES — checked here so
 *  the user gets an instant answer instead of a websocket round-trip. */
export const MAX_BYTES = 20 * 1024 * 1024;

export function oversized(files: File[]): string[] {
  return files.filter((f) => f.size > MAX_BYTES).map((f) => f.name);
}

function base64(bytes: Uint8Array): string {
  let binary = "";
  const CHUNK = 0x8000; // keep the spread call under the argument limit
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/** Path-basename for messages (the sidecar reports its own, this is for the
 *  client-side "already imported" checks and the confirm dialog). */
export function baseName(path: string): string {
  const norm = path.replace(/\\/g, "/");
  return norm.slice(norm.lastIndexOf("/") + 1) || path;
}
