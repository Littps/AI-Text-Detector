// WS client for the sidecar protocol. Request/response correlation via
// auto-increment ids; server events ("event" envelopes) fan out to
// subscribers. No framework coupling - one class, plain callbacks.
import type { DlEvent, Envelope } from "./types";

export interface ClientOptions {
  port: number;
  token: string;
  /** Reconnect backoff base, ms (default 1000). */
  backoffMs?: number;
  /** Request timeout, ms (default 120000) - guards against a wedged
   * sidecar leaving pending promises unresolved forever. */
  requestTimeoutMs?: number;
}

export interface RequestOptions {
  /**
   * 0 disables the client-side timeout. Detection is intentionally a long
   * request: the backend owns cancellation and progress, while a fixed
   * client timer would leave the GPU job running after a false timeout.
   */
  timeoutMs?: number;
}

type EventListener = (ev: DlEvent) => void;
type StateListener = (state: "connecting" | "open" | "closed") => void;

export class SidecarClient {
  private ws: WebSocket | null = null;
  private seq = 0;
  private pending = new Map<string, { resolve: (p: unknown) => void; reject: (e: Error) => void }>();
  private eventListeners = new Set<EventListener>();
  private stateListeners = new Set<StateListener>();
  private opts: ClientOptions;
  private shouldReconnect = false;
  private backoffTimer: number | undefined;

  constructor(opts: ClientOptions) {
    this.opts = opts;
  }

  url(): string {
    return `ws://127.0.0.1:${this.opts.port}/?token=${encodeURIComponent(this.opts.token)}`;
  }

  onEvent(cb: EventListener): () => void {
    this.eventListeners.add(cb);
    return () => this.eventListeners.delete(cb);
  }

  onState(cb: StateListener): () => void {
    this.stateListeners.add(cb);
    return () => this.stateListeners.delete(cb);
  }

  private emitState(s: "connecting" | "open" | "closed") {
    for (const cb of this.stateListeners) cb(s);
  }

  connect(): void {
    this.shouldReconnect = true;
    // Idempotent: one client owns one socket. Re-running the mount effect
    // (StrictMode, HMR) used to stack a second socket on top of the first;
    // the sidecar's single-client rule then rejected every retry while the
    // orphan held the slot (round-28 "backend closed" loop).
    if (this.isLive()) return;
    this.open();
  }

  private isLive(): boolean {
    const rs = this.ws?.readyState;
    return rs === WebSocket.OPEN || rs === WebSocket.CONNECTING;
  }

  private open() {
    if (this.isLive()) return;
    this.emitState("connecting");
    const ws = new WebSocket(this.url());
    this.ws = ws;

    ws.onopen = () => {
      if (this.ws === ws) this.emitState("open");
    };
    ws.onmessage = (ev) => this.dispatch(ev.data as string);
    ws.onclose = () => {
      // a socket we already replaced must not flip the new client's state
      if (this.ws !== ws) return;
      this.ws = null;
      this.failPending(new Error("connection closed"));
      this.emitState("closed");
      if (this.shouldReconnect) {
        this.backoffTimer = window.setTimeout(() => this.open(), this.opts.backoffMs ?? 1000);
      }
    };
    ws.onerror = () => ws.close();
  }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.backoffTimer !== undefined) {
      window.clearTimeout(this.backoffTimer);
      this.backoffTimer = undefined;
    }
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      // Detach before closing: a replaced client's onclose would otherwise
      // emit "closed" *after* the replacement went "open".
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      ws.close();
    }
    this.failPending(new Error("client disconnected"));
  }

  private failPending(err: Error) {
    for (const [, p] of this.pending) p.reject(err);
    this.pending.clear();
  }

  private dispatch(raw: string) {
    let env: Envelope;
    try {
      env = JSON.parse(raw);
    } catch {
      return; // malformed frame - drop silently
    }
    if (env.type === "event") {
      for (const cb of this.eventListeners) cb(env.payload as DlEvent);
      return;
    }
    if (env.type === "res") {
      const p = this.pending.get(env.id);
      if (!p) return;
      this.pending.delete(env.id);
      const payload = env.payload as { error?: string };
      if (payload && typeof payload.error === "string") {
        p.reject(new Error(payload.error));
      } else {
        p.resolve(env.payload);
      }
    }
  }

  /** Send a request and await the correlated "res" payload. */
  request<T>(
    type: string,
    payload?: Record<string, unknown>,
    options?: RequestOptions,
  ): Promise<T> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("not connected"));
    }
    this.seq += 1;
    const id = `m-${this.seq}`;
    const env: Envelope = { v: 1, id, type, ts: Date.now(), payload: payload ?? {} };
    return new Promise<T>((resolve, reject) => {
      const timeoutMs =
        options?.timeoutMs ?? (type === "detect" ? 0 : this.opts.requestTimeoutMs ?? 120_000);
      let timer: number | undefined;
      if (timeoutMs > 0) {
        timer = window.setTimeout(() => {
          this.pending.delete(id);
          reject(new Error(`request ${type} timed out`));
        }, timeoutMs);
      }
      this.pending.set(id, {
        resolve: (p: unknown) => {
          if (timer !== undefined) window.clearTimeout(timer);
          resolve(p as T);
        },
        reject: (e: Error) => {
          if (timer !== undefined) window.clearTimeout(timer);
          reject(e);
        },
      });
      ws.send(JSON.stringify(env));
    });
  }
}

export interface ProgressFrame {
  done: number;
  total: number;
  dest: string;
}

export function fmtBytes(n: number): string {
  if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(2)} GB`;
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MB`;
  if (n >= 1 << 10) return `${(n / (1 << 10)).toFixed(1)} KB`;
  return `${n} B`;
}

/** MB -> human string with the app's 1024-based convention (7.93 GB card). */
export function fmtMb(mb: number): string {
  if (!Number.isFinite(mb) || mb <= 0) return "0 MB";
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${Math.round(mb)} MB`;
}
