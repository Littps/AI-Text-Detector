// D25 storage panel: the three-mode switch per engine + cast/cleanup
// actions. Destructive-ish actions (cast overwrites fp32 in fp16-only;
// cleanup deletes copies) show explicit confirmation text inline.
import { useEffect, useRef, useState } from "react";
import {
  STORAGE_MODE_LABEL,
  storageBadge,
} from "./highlight";
import {
  STORAGE_MODES,
  type CleanupPayload,
  type DlEngineStatus,
  type SetModePayload,
  type StorageMode,
  type StorageUninstallPayload,
} from "./types";
import type { SidecarClient, ProgressFrame } from "./ws";
import { fmtBytes } from "./ws";
import { Modal } from "./Modal";
import { SPRING_CLOSE } from "./spring";

interface Props {
  client: SidecarClient | null;
  engines: [string, DlEngineStatus][];
  /** friendly display names by engine id (round-6: no raw ids in tables) */
  names?: Record<string, string>;
  onDone: () => void; // refresh statuses
  castProgress: Record<string, ProgressFrame>;
  onError: (msg: string) => void;
  /** success/status line (round-25: cleanup results must not turn the health
   * light red - ``onError`` is wired to ``deriveHealth``) */
  onNotice: (msg: string) => void;
  jobRunning: boolean;
}

export function StoragePanel({ client, engines, names, onDone, castProgress, onError, onNotice, jobRunning }: Props) {
  const [confirming, setConfirming] = useState<{
    eid: string;
    action: "set" | "cleanup" | "uninstall";
    mode?: StorageMode;
  } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmClosing, setConfirmClosing] = useState(false);
  const closeTimer = useRef<number | undefined>(undefined);

  useEffect(
    () => () => {
      window.clearTimeout(closeTimer.current);
    },
    [],
  );

  const openConfirm = (next: NonNullable<typeof confirming>) => {
    window.clearTimeout(closeTimer.current);
    setConfirmClosing(false);
    setConfirming(next);
  };

  const beginClose = () => {
    if (confirmClosing) return;
    setConfirmClosing(true);
    closeTimer.current = window.setTimeout(() => {
      setConfirming(null);
      setConfirmClosing(false);
    }, SPRING_CLOSE.duration);
  };

  const requestClose = () => {
    if (busy === null) beginClose();
  };

  const apply = async (eid: string, mode: StorageMode) => {
    if (!client) return;
    setBusy(eid);
    try {
      await client.request<SetModePayload>("storage/set_mode", { engine_id: eid, mode });
      onDone();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(null);
    }
  };

  const cast = async (eid: string) => {
    if (!client) return;
    setBusy(eid);
    try {
      await client.request("storage/cast", { engine_id: eid });
      onDone();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(null);
    }
  };

  const cleanup = async (eid: string) => {
    if (!client) return;
    setBusy(eid);
    try {
      const res = await client.request<CleanupPayload>("storage/cleanup", { engine_id: eid });
      onNotice(`已清理 ${res.removed.length} 个副本文件`);
      onDone();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(null);
    }
  };

  const uninstall = async (eid: string) => {
    if (!client) return;
    setBusy(eid);
    try {
      const res = await client.request<StorageUninstallPayload>("storage/uninstall", {
        engine_id: eid,
      });
      const leftover =
        res.leftovers.length > 0 ? `，保留 ${res.leftovers.length} 个未登记文件` : "";
      onNotice(
        `已卸载 ${names?.[eid] ?? eid} · 释放 ${fmtBytes(res.bytes_freed)}${leftover}`,
      );
      onDone();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(null);
    }
  };

  const confirmRow = confirming
    ? engines.find(([eid]) => eid === confirming.eid)
    : undefined;
  const confirmName = confirming
    ? names?.[confirming.eid] ?? confirming.eid
    : "";

  const confirmAction = async () => {
    if (!confirming) return;
    if (confirming.action === "set") {
      await apply(confirming.eid, StorageModeOk(confirming.mode));
    } else if (confirming.action === "cleanup") {
      await cleanup(confirming.eid);
    } else {
      await uninstall(confirming.eid);
    }
    beginClose();
  };

  if (engines.length === 0) return null;
  return (
    <>
      <div className="storage-panel panel elevated">
        <h3>存储管理</h3>
        <p className="hint">
          每个模型可独立选择三种保存方式：
          <strong>保留原件</strong>——只存 fp32 原件，检测时现场临时转半精度（最省心，占盘最多）；
          <strong>双份</strong>——额外保存一个 fp16 副本，检测直接加载副本、更快就绪；
          <strong>仅半精度</strong>——转换后覆盖原件，磁盘占用约省一半，但 fp32 原件不再直接可用
          （切回「保留原件」需要重新下载）。操作只影响存储，不改变检测结果。
        </p>
        <table>
          <thead>
            <tr>
              <th>引擎</th>
              <th>磁盘状态</th>
              <th>存储方式</th>
              <th>操作</th>
              <th>转换进度</th>
            </tr>
          </thead>
          <tbody>
            {engines.map(([eid, s]) => {
              const st = s.storage;
              const prog = castProgress[eid];
              return (
                <tr key={eid}>
                  <td>
                    <div>{names?.[eid] ?? eid}</div>
                    {names?.[eid] && <div className="eng-repo">{eid}</div>}
                  </td>
                  <td>
                    <span className="badge">{storageBadge(st)}</span>
                    {st.recast_needed && <span className="badge warn">待转换</span>}
                    {st.stale_cast && <span className="badge warn">副本多余</span>}
                  </td>
                  <td>
                    <select
                      value={st.storage_mode}
                      disabled={!client || busy === eid || jobRunning}
                      onChange={(e) => {
                        const mode = e.target.value as StorageMode;
                        if (mode === "fp16-only" && st.fp32_present) {
                          openConfirm({ eid, action: "set", mode });
                        } else {
                          void apply(eid, mode);
                        }
                      }}
                    >
                      {STORAGE_MODES.map((m) => (
                        <option key={m} value={m}>
                          {STORAGE_MODE_LABEL[m]}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <div className="storage-actions">
                      {st.recast_needed && (
                        <button
                          disabled={!client || busy === eid || jobRunning}
                          onClick={() => void cast(eid)}
                        >
                          转换
                        </button>
                      )}
                      {st.stale_cast && (
                        <button
                          disabled={!client || busy === eid || jobRunning}
                          onClick={() => openConfirm({ eid, action: "cleanup" })}
                        >
                          清理副本
                        </button>
                      )}
                      {st.installed && (
                        <button
                          className="danger-ghost"
                          disabled={!client || busy === eid || jobRunning}
                          onClick={() => openConfirm({ eid, action: "uninstall" })}
                        >
                          卸载模型
                        </button>
                      )}
                    </div>
                  </td>
                  <td>
                    {prog ? `${fmtBytes(prog.done)} / ${fmtBytes(prog.total)}` : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <Modal
        open={confirming !== null}
        closing={confirmClosing}
        onClose={requestClose}
        className="storage-confirm-modal"
        ariaLabel="存储操作确认"
      >
        <h3>
          {confirming?.action === "uninstall"
            ? "卸载模型"
            : confirming?.action === "cleanup"
              ? "清理转换副本"
              : "切换为仅半精度"}
        </h3>
        <p>
          {confirming?.action === "uninstall" && (
            <>
              将从磁盘删除 <b>{confirmName}</b> 的已登记权重、配置、转换副本和未完成下载。
              之后重新检测需要再次下载；不会删除其他模型或自定义库。
            </>
          )}
          {confirming?.action === "cleanup" && (
            <>
              将删除 <b>{confirmName}</b> 的 fp16 副本与转换索引，保留 fp32 原件和检测结果能力。
            </>
          )}
          {confirming?.action === "set" && (
            <>
              <b>{confirmName}</b> 将切换为仅半精度：转换完成后覆盖 fp32 原件，
              切回保留原件需要重新下载。
            </>
          )}
        </p>
        {confirming?.action === "uninstall" && confirmRow && (
          <p className="hint">
            当前登记体积约 {fmtBytes(confirmRow[1].total_bytes)}。卸载前会先释放该模型占用的显存。
          </p>
        )}
        <div className="modal-actions">
          <button onClick={requestClose} disabled={busy !== null || confirmClosing}>
            取消
          </button>
          <button
            className={confirming?.action === "uninstall" ? "danger" : "primary"}
            onClick={() => void confirmAction()}
            disabled={busy !== null}
          >
            {busy === confirming?.eid ? "处理中…" : "确认"}
          </button>
        </div>
      </Modal>
    </>
  );
}

// tiny helper so the JSX above stays readable
function StorageModeOk(mode: StorageMode | undefined): StorageMode {
  return mode ?? "fp16-only";
}
