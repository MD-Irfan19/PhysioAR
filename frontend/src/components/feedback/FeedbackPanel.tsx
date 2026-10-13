import React from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  Info,
  UserX,
  FileText,
} from 'lucide-react'
import type { SessionSnapshot } from '@/types'

interface FeedbackPanelProps {
  snapshot: SessionSnapshot | null
}

const COMPENSATION_NAMES: Record<string, string> = {
  torso_lean: 'Torso Lateral Lean',
  shoulder_hike: 'Shoulder Hike / Elevation',
  neck_tilt: 'Cervical Spine Tilt',
  hip_rotation: 'Pelvic / Hip Rotation',
  lateral_trunk_lean: 'Trunk Lateral Deviation',
  hip_hike: 'Pelvic Hike',
  shoulder_substitution: 'Shoulder Substitution',
  insufficient_forward_trunk_lean: 'Insufficient Forward Trunk Inclination',
  hand_assisted_rising: 'Hand-Assisted Rising Proxy',
  asymmetric_weight_shift: 'Asymmetric Weight Shift Proxy',
}

export const FeedbackPanel: React.FC<FeedbackPanelProps> = ({ snapshot }) => {
  const feedbackMessages = snapshot?.feedback_messages ?? []
  const activeCompensations = snapshot?.active_compensation_ids ?? []
  const poseDetected = snapshot?.pose_detected ?? false
  const isRunning =
    snapshot?.status === 'active' ||
    snapshot?.status === 'calibrating' ||
    snapshot?.status === 'error'
  const calibrationError = snapshot?.calibration_error

  return (
    <div className="rounded-xl border border-border bg-panel p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between border-b border-border pb-3">
        <h2 className="text-base font-semibold text-foreground flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-400" />
          Real-Time Corrective Feedback
        </h2>
        {activeCompensations.length > 0 && (
          <span className="rounded bg-status-warning/15 px-2 py-0.5 text-xs font-semibold text-status-warning border border-status-warning/30">
            {activeCompensations.length} Flagged
          </span>
        )}
      </div>

      {/* Subject Not Detected Warning */}
      {isRunning && !poseDetected && (
        <div className="flex items-start gap-2.5 rounded-lg border border-status-error/40 bg-status-error/10 p-3 text-xs text-status-error">
          <UserX className="h-4 w-4 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Subject Not Detected</p>
            <p className="mt-0.5 text-foreground-muted">
              Step back into the camera frame. Ensure adequate lighting and full-body visibility.
            </p>
          </div>
        </div>
      )}

      {/* Calibration Error Notification */}
      {calibrationError && (
        <div className="flex items-start gap-2.5 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-400">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Calibration Baseline Warning</p>
            <p className="mt-0.5 text-foreground-muted">{calibrationError}</p>
          </div>
        </div>
      )}

      {/* Active Compensations List */}
      {activeCompensations.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold text-foreground-muted uppercase tracking-wider">
            Flagged Compensatory Movements
          </h3>
          <div className="flex flex-wrap gap-2">
            {activeCompensations.map((id) => (
              <span
                key={id}
                className="inline-flex items-center gap-1.5 rounded-lg border border-status-error/30 bg-status-error/10 px-3 py-1.5 text-xs font-medium text-status-error shadow-sm"
              >
                <ShieldAlert className="h-3.5 w-3.5" />
                <span>{COMPENSATION_NAMES[id] || id.replace(/_/g, ' ')}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Form Cue Messages */}
      {feedbackMessages.length > 0 ? (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold text-foreground-muted uppercase tracking-wider">
            Corrective Guidance
          </h3>
          <ul className="space-y-2">
            {feedbackMessages.map((msg, idx) => (
              <li
                key={idx}
                className="flex items-start gap-2 rounded-lg border border-amber-500/25 bg-amber-500/10 p-3 text-xs font-medium text-amber-300"
              >
                <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
                <span className="leading-relaxed">{msg}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : isRunning && poseDetected && activeCompensations.length === 0 ? (
        <div className="flex items-center gap-2.5 rounded-lg border border-status-success/20 bg-status-success/5 p-3.5 text-xs text-status-success">
          <CheckCircle2 className="h-4 w-4 shrink-0" />
          <span>Optimal form maintained — no compensatory movements flagged.</span>
        </div>
      ) : !isRunning ? (
        <div className="flex items-center gap-2.5 rounded-lg border border-border bg-panel-subtle p-3.5 text-xs text-foreground-muted">
          <Info className="h-4 w-4 shrink-0 text-foreground-subtle" />
          <span>Start an exercise session to receive real-time corrective cues.</span>
        </div>
      ) : null}

      {/* Clinical Assessment Disclaimer */}
      <div className="rounded-lg border border-border/80 bg-panel-subtle/60 p-3 text-[11px] text-foreground-subtle space-y-1">
        <div className="flex items-center gap-1.5 font-semibold text-foreground-muted">
          <FileText className="h-3 w-3 text-accent" />
          <span>Clinical Disclaimer</span>
        </div>
        <p className="leading-relaxed">
          PhysioAR is an engineering guidance tool based on 2D MediaPipe landmarks. It does not
          diagnose musculoskeletal pathology, guarantee clinical correctness, or substitute for licensed physiotherapy care.
        </p>
      </div>
    </div>
  )
}
