import React from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  Info,
  UserX,
  FileText,
  Activity,
  Compass,
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
  const status = snapshot?.status ?? 'idle'
  const isRunning =
    status === 'active' || status === 'calibrating' || status === 'error'
  const poseDetected = Boolean(snapshot?.pose_detected)
  const feedbackMessages = Array.isArray(snapshot?.feedback_messages)
    ? snapshot.feedback_messages.filter(Boolean)
    : []
  const activeCompensations = Array.isArray(snapshot?.active_compensation_ids)
    ? snapshot.active_compensation_ids.filter(Boolean)
    : []
  const calibrationError =
    typeof snapshot?.calibration_error === 'string'
      ? snapshot.calibration_error
      : null

  const hasCompensations = activeCompensations.length > 0
  const hasFeedback = feedbackMessages.length > 0
  const isActivelyTracking = status === 'active' && poseDetected
  const isCalibrating = status === 'calibrating'

  return (
    <div
      data-testid="feedback-panel"
      aria-label="Real-Time Corrective Feedback Panel"
      className="rounded-xl border border-border bg-panel p-4 sm:p-5 shadow-sm space-y-3.5 sm:space-y-4 w-full min-w-0"
    >
      {/* 1. Header & Live State Badge */}
      <div className="flex items-center justify-between border-b border-border pb-3">
        <h2 className="text-sm sm:text-base font-semibold text-foreground flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-400" aria-hidden="true" />
          Real-Time Corrective Feedback
        </h2>
        {hasCompensations ? (
          <span className="rounded-full bg-status-warning/15 px-2.5 py-0.5 text-xs font-semibold text-status-warning border border-status-warning/30 flex items-center gap-1">
            <ShieldAlert className="h-3 w-3" aria-hidden="true" />
            {activeCompensations.length} Flagged
          </span>
        ) : isRunning && !poseDetected ? (
          <span className="rounded-full bg-status-error/15 px-2 py-0.5 text-xs font-semibold text-status-error border border-status-error/30">
            No Pose
          </span>
        ) : isActivelyTracking ? (
          <span className="rounded-full bg-status-success/15 px-2 py-0.5 text-xs font-semibold text-status-success border border-status-success/30 flex items-center gap-1">
            <CheckCircle2 className="h-3 w-3" aria-hidden="true" />
            Aligned
          </span>
        ) : isCalibrating ? (
          <span className="rounded-full bg-amber-400/15 px-2 py-0.5 text-xs font-semibold text-amber-400 border border-amber-400/30">
            Calibrating
          </span>
        ) : (
          <span className="rounded-full bg-panel-subtle px-2 py-0.5 text-xs font-medium text-foreground-subtle border border-border">
            Standby
          </span>
        )}
      </div>

      {/* 2. Critical Pose Visibility Warning */}
      {isRunning && !poseDetected && (
        <div
          role="alert"
          className="flex items-start gap-2.5 rounded-lg border border-status-error/40 bg-status-error/10 p-3 text-xs text-status-error shadow-sm"
        >
          <UserX className="h-4 w-4 shrink-0 mt-0.5" aria-hidden="true" />
          <div className="space-y-0.5 min-w-0">
            <p className="font-semibold">Subject Not Detected</p>
            <p className="text-foreground-muted leading-relaxed">
              Step back into the camera frame. Ensure adequate lighting and full-body visibility.
            </p>
          </div>
        </div>
      )}

      {/* 3. Calibration Warning / Baseline Status */}
      {calibrationError && (
        <div
          role="alert"
          className="flex items-start gap-2.5 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-400 shadow-sm"
        >
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-amber-400" aria-hidden="true" />
          <div className="space-y-0.5 min-w-0">
            <p className="font-semibold">Calibration Baseline Warning</p>
            <p className="text-foreground-muted leading-relaxed">{calibrationError}</p>
          </div>
        </div>
      )}

      {/* 4. Active Compensations (Movement Warning Priority) */}
      {hasCompensations && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-[11px] font-semibold text-status-warning uppercase tracking-wider flex items-center gap-1.5">
              <ShieldAlert className="h-3.5 w-3.5 text-status-warning" aria-hidden="true" />
              Flagged Compensatory Movements
            </h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {activeCompensations.map((id) => {
              const label =
                COMPENSATION_NAMES[id] ||
                id.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
              return (
                <span
                  key={id}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-status-warning/40 bg-status-warning/15 px-3 py-1.5 text-xs font-semibold text-amber-300 shadow-sm"
                >
                  <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-status-warning" aria-hidden="true" />
                  <span>{label}</span>
                </span>
              )
            })}
          </div>
        </div>
      )}

      {/* 5. Corrective Action Guidance (Actionable Cues) */}
      {hasFeedback && (
        <div className="space-y-2">
          <h3 className="text-[11px] font-semibold text-accent uppercase tracking-wider flex items-center gap-1.5">
            <Activity className="h-3.5 w-3.5 text-accent" aria-hidden="true" />
            Corrective Guidance
          </h3>
          <ul className="space-y-2 max-h-56 overflow-y-auto pr-1">
            {feedbackMessages.map((msg, idx) => (
              <li
                key={idx}
                className="flex items-start gap-2.5 rounded-lg border border-accent/30 bg-accent/10 p-3 text-xs font-medium text-foreground shadow-sm"
              >
                <span
                  aria-hidden="true"
                  className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent/20 text-accent font-bold text-[11px]"
                >
                  {idx + 1}
                </span>
                <span className="leading-relaxed mt-0.5">{msg}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 6. Form Alignment Confirmation (Restrained Success) */}
      {isActivelyTracking && !hasCompensations && !hasFeedback && (
        <div className="flex items-start gap-2.5 rounded-lg border border-status-success/30 bg-status-success/10 p-3 text-xs text-status-success shadow-sm">
          <CheckCircle2 className="h-4 w-4 shrink-0 mt-0.5 text-status-success" aria-hidden="true" />
          <div className="space-y-0.5 min-w-0">
            <p className="font-semibold text-status-success">Optimal Form</p>
            <p className="text-foreground-muted text-[11px] leading-relaxed">
              Optimal form maintained — no compensatory movements flagged.
            </p>
          </div>
        </div>
      )}

      {/* 7. Calibrating State Guidance */}
      {isCalibrating && !hasCompensations && !calibrationError && (
        <div className="flex items-start gap-2.5 rounded-lg border border-accent/25 bg-accent/5 p-3 text-xs text-accent">
          <Compass className="h-4 w-4 shrink-0 mt-0.5 text-accent animate-pulse" aria-hidden="true" />
          <div className="space-y-0.5 min-w-0">
            <p className="font-semibold text-foreground">Neutral Baseline Calibration</p>
            <p className="text-foreground-muted text-[11px] leading-relaxed">
              Stand in neutral upright posture facing the camera while baseline landmarks are collected.
            </p>
          </div>
        </div>
      )}

      {/* 8. Standby / Idle State Guidance */}
      {!isRunning && (
        <div className="flex items-start gap-2.5 rounded-lg border border-border bg-panel-subtle/80 p-3 text-xs text-foreground-muted">
          <Info className="h-4 w-4 shrink-0 mt-0.5 text-foreground-subtle" aria-hidden="true" />
          <div className="space-y-0.5 min-w-0">
            <p className="font-semibold text-foreground">Session Standby</p>
            <p className="text-foreground-subtle text-[11px] leading-relaxed">
              Start an exercise session to receive real-time corrective cues.
            </p>
          </div>
        </div>
      )}

      {/* 9. Clinical Disclaimer (De-emphasized Compact Section) */}
      <div className="rounded-lg border border-border/70 bg-panel-subtle/50 p-2.5 text-[10px] sm:text-[11px] text-foreground-subtle space-y-1">
        <div className="flex items-center gap-1.5 font-semibold text-foreground-muted">
          <FileText className="h-3 w-3 text-accent shrink-0" aria-hidden="true" />
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
