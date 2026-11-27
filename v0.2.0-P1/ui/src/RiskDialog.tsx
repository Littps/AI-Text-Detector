// D20 risk double-dialog (PLAN §3 P2): never locks options - only gates
// checkbox commitment. Step 1 states the risk; step 2 holds a 5s
// countdown so the user has read it before the confirm button unlocks.
import { useEffect, useState } from "react";

export const RISK_TEXT: Record<string, string> = {
  "risk.fdg.exclusive":
    "Fast-DetectGPT 以独占模式运行（fp16 驻留约 5.4GB 显存）：启用时其余 GPU 引擎将被卸载，检测延迟也明显高于其他引擎（每句 25 次采样）。显存不足 8GB 的设备可能失败。",
  "risk.binoculars.8g":
    "Binoculars 结构性绑定双 7B 模型（下载约 28.9GB，fp16 驻留约 28GB，独占运行）：显存不足 28GB 的设备会直接 OOM——本机 8GB 显卡无法运行。4bit 量化路径未实现（缺少 bitsandbytes 的 cp314 轮子），下载权重也无法在 8GB 卡上跑起来。",
  default:
    "该引擎为高风险引擎：显存占用高或与其他引擎互斥，启用前请确认设备资源充足。",
};

interface Props {
  engineId: string;
  riskLevel: string;
  riskTextKey: string | null;
  onConfirmed: () => void;
  onCancelled: () => void;
}

export function RiskDialog({
  engineId,
  riskLevel,
  riskTextKey,
  onConfirmed,
  onCancelled,
}: Props) {
  const [step, setStep] = useState<1 | 2>(1);
  const [left, setLeft] = useState(5);

  useEffect(() => {
    if (step !== 2) return;
    if (left <= 0) return;
    const t = window.setTimeout(() => setLeft((x) => x - 1), 1000);
    return () => window.clearTimeout(t);
  }, [step, left]);

  const text =
    (riskTextKey && RISK_TEXT[riskTextKey]) || RISK_TEXT.default;

  return (
    <div className="modal-backdrop">
      <div className="modal glass">
        <h3>
          风险确认（{step}/2）— {engineId} [{riskLevel}]
        </h3>
        <p className="risk-text">{text}</p>
        {step === 1 ? (
          <div className="row">
            <button onClick={() => setStep(2)}>我已了解，继续</button>
            <button onClick={onCancelled}>取消</button>
          </div>
        ) : (
          <div className="row">
            <button disabled={left > 0} onClick={onConfirmed}>
              {left > 0 ? `确认启用（${left}s）` : "确认启用"}
            </button>
            <button onClick={onCancelled}>取消</button>
          </div>
        )}
      </div>
    </div>
  );
}
