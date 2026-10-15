import React from 'react'
import {
  RotateCcw,
  Gauge,
  Target,
  Compass,
} from 'lucide-react'
import type { SessionSnapshot } from '@/types'

interface MetricCardsProps {
  snapshot: SessionSnapshot | null
}

export const MetricCards: React.FC<MetricCardsProps> = ({ snapshot }) => {
  // 1. Joint Angle (degrees)
  const rawAngle = snapshot?.current_angle ?? snapshot?.joint_angle
  const currentAngle =
    typeof rawAngle === 'number' && Number.isFinite(rawAngle) ? rawAngle : null

  // Target Angle (degrees)
  const rawTarget = snapshot?.rom_target
  const romTarget =
    typeof rawTarget === 'number' && Number.isFinite(rawTarget) ? rawTarget : null

  // 2. ROM Progress (0.0 to 1.0 ratio -> 0% to 100%)
  const rawProgress = snapshot?.rom_progress
  const progressPercent =
    typeof rawProgress === 'number' && Number.isFinite(rawProgress)
      ? Math.min(Math.max(Math.round(rawProgress * 100), 0), 100)
      : null

  // 3. Repetitions
  const rawReps = snapshot?.rep_count
  const repCount =
    typeof rawReps === 'number' && Number.isFinite(rawReps)
      ? Math.max(0, Math.floor(rawReps))
      : 0

  const repState =
    typeof snapshot?.rep_state === 'string'
      ? snapshot.rep_state
      : typeof snapshot?.rep_phase === 'string'
      ? snapshot.rep_phase
      : null

  // 4. Calibration Status
  const isCalibrated = Boolean(snapshot?.is_calibrated)
  const calibStatus =
    typeof snapshot?.calibration_status === 'string'
      ? snapshot.calibration_status
      : 'not_calibrated'

  const rawSamples = snapshot?.calibration_samples
  const calibSamples =
    typeof rawSamples === 'number' && Number.isFinite(rawSamples)
      ? Math.max(0, Math.floor(rawSamples))
      : 0

  const calibError =
    typeof snapshot?.calibration_error === 'string'
      ? snapshot.calibration_error
      : null

  return (
    <div
      data-testid="telemetry-strip"
      aria-label="Real-time exercise telemetry"
      className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-2.5 w-full min-w-0"
    >
      {/* 1. Joint Angle */}
      <div className="rounded-lg sm:rounded-xl border border-border bg-panel/90 px-3 py-2 sm:px-3.5 sm:py-2.5 shadow-sm flex flex-col justify-between min-w-0 overflow-hidden">
        <div className="flex items-center justify-between text-foreground-muted mb-1">
          <span className="text-[10px] sm:text-[11px] font-semibold uppercase tracking-wider truncate">
            Joint Angle
          </span>
          <Gauge className="h-3.5 w-3.5 shrink-0 text-accent" aria-hidden="true" />
        </div>
        <div className="min-w-0">
          <div className="flex items-baseline gap-0.5 min-w-0">
            <span className="text-xl sm:text-2xl font-bold font-mono text-foreground tracking-tight truncate">
              {currentAngle !== null ? currentAngle.toFixed(1) : '—'}
            </span>
            {currentAngle !== null && (
              <span className="text-xs font-semibold text-accent ml-0.5">°</span>
            )}
          </div>
          <p className="text-[10px] sm:text-[11px] text-foreground-subtle truncate mt-0.5">
            {romTarget !== null ? `Target: ${romTarget}°` : 'Biomechanical angle'}
          </p>
        </div>
      </div>

      {/* 2. Repetition Counter */}
      <div className="rounded-lg sm:rounded-xl border border-border bg-panel/90 px-3 py-2 sm:px-3.5 sm:py-2.5 shadow-sm flex flex-col justify-between min-w-0 overflow-hidden">
        <div className="flex items-center justify-between text-foreground-muted mb-1">
          <span className="text-[10px] sm:text-[11px] font-semibold uppercase tracking-wider truncate">
            Repetitions
          </span>
          <RotateCcw className="h-3.5 w-3.5 shrink-0 text-emerald-400" aria-hidden="true" />
        </div>
        <div className="min-w-0">
          <div className="flex items-baseline gap-1.5 min-w-0">
            <span className="text-xl sm:text-2xl font-bold font-mono text-emerald-400 tracking-tight truncate">
              {repCount}
            </span>
            <span className="text-[10px] sm:text-xs font-medium text-foreground-muted truncate">
              Reps Completed
            </span>
          </div>
          <p className="text-[10px] sm:text-[11px] text-foreground-subtle truncate mt-0.5 capitalize">
            State: {repState ? repState.replace(/_/g, ' ') : 'Waiting for motion'}
          </p>
        </div>
      </div>

      {/* 3. Range of Motion (ROM) Progress */}
      <div className="rounded-lg sm:rounded-xl border border-border bg-panel/90 px-3 py-2 sm:px-3.5 sm:py-2.5 shadow-sm flex flex-col justify-between min-w-0 overflow-hidden">
        <div className="flex items-center justify-between text-foreground-muted mb-1">
          <span className="text-[10px] sm:text-[11px] font-semibold uppercase tracking-wider truncate">
            ROM Progress
          </span>
          <Target className="h-3.5 w-3.5 shrink-0 text-amber-400" aria-hidden="true" />
        </div>
        <div className="min-w-0">
          <div className="flex items-baseline justify-between gap-1 min-w-0">
            <span className="text-xl sm:text-2xl font-bold font-mono text-amber-400 tracking-tight truncate">
              {progressPercent !== null ? `${progressPercent}%` : '—'}
            </span>
            {progressPercent !== null && (
              <span className="text-[10px] text-foreground-subtle font-mono hidden sm:inline">
                {progressPercent >= 100 ? 'Target met' : `${100 - progressPercent}% to target`}
              </span>
            )}
          </div>
          {/* Compact Progress Bar */}
          <div
            className="w-full bg-panel-hover rounded-full h-1 mt-1 overflow-hidden"
            role="progressbar"
            aria-valuenow={progressPercent ?? 0}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="ROM Completion Percentage"
          >
            <div
              className="bg-amber-400 h-1 rounded-full transition-all duration-150"
              style={{ width: `${progressPercent ?? 0}%` }}
            />
          </div>
        </div>
      </div>

      {/* 4. Calibration Status */}
      <div className="rounded-lg sm:rounded-xl border border-border bg-panel/90 px-3 py-2 sm:px-3.5 sm:py-2.5 shadow-sm flex flex-col justify-between min-w-0 overflow-hidden">
        <div className="flex items-center justify-between text-foreground-muted mb-1">
          <span className="text-[10px] sm:text-[11px] font-semibold uppercase tracking-wider truncate">
            Calibration
          </span>
          <Compass className="h-3.5 w-3.5 shrink-0 text-accent" aria-hidden="true" />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 min-w-0">
            <span
              aria-hidden="true"
              className={`h-2 w-2 shrink-0 rounded-full ${
                isCalibrated
                  ? 'bg-status-success'
                  : calibStatus === 'calibrating'
                  ? 'bg-amber-400 animate-ping'
                  : calibStatus === 'failed'
                  ? 'bg-status-error'
                  : 'bg-foreground-subtle'
              }`}
            />
            <span className="text-sm sm:text-base font-bold text-foreground truncate">
              {calibStatus === 'calibrated'
                ? 'Baseline Active'
                : calibStatus === 'calibrating'
                ? 'Collecting...'
                : calibStatus === 'failed'
                ? 'Calibration Failed'
                : 'Uncalibrated'}
            </span>
          </div>
          <p className="text-[10px] sm:text-[11px] text-foreground-subtle truncate mt-0.5">
            {calibStatus === 'failed'
              ? calibError || 'Insufficient samples'
              : calibSamples > 0
              ? `${calibSamples} valid samples`
              : 'Awaiting baseline'}
          </p>
        </div>
      </div>
    </div>
  )
}
