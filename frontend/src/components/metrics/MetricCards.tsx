import React from 'react'
import {
  RotateCcw,
  Gauge,
  Target,
  Sliders,
  Compass,
  TrendingUp,
} from 'lucide-react'
import type { SessionSnapshot } from '@/types'

interface MetricCardsProps {
  snapshot: SessionSnapshot | null
}

export const MetricCards: React.FC<MetricCardsProps> = ({ snapshot }) => {
  const rawAngle = snapshot?.current_angle ?? snapshot?.joint_angle
  const currentAngle =
    typeof rawAngle === 'number' && !isNaN(rawAngle) ? rawAngle : null
  const romTarget = snapshot?.rom_target
  const romProgress = snapshot?.rom_progress
  const repCount = snapshot?.rep_count ?? 0
  const repState = typeof snapshot?.rep_state === 'string'
    ? snapshot.rep_state
    : typeof snapshot?.rep_phase === 'string'
    ? snapshot.rep_phase
    : null
  const isCalibrated = snapshot?.is_calibrated ?? false
  const calibStatus = snapshot?.calibration_status ?? 'not_calibrated'
  const calibSamples = snapshot?.calibration_samples ?? 0

  // Calculate percentage safely
  const progressPercent =
    typeof romProgress === 'number' && !isNaN(romProgress)
      ? Math.min(Math.max(Math.round(romProgress * 100), 0), 100)
      : null

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {/* 1. Joint Angle */}
      <div className="rounded-xl border border-border bg-panel p-4 shadow-sm flex flex-col justify-between">
        <div className="flex items-center justify-between text-foreground-muted mb-2">
          <span className="text-xs font-medium uppercase tracking-wider">Joint Angle</span>
          <Gauge className="h-4 w-4 text-accent" />
        </div>
        <div>
          <div className="flex items-baseline gap-1">
            <span className="text-3xl font-extrabold text-foreground font-mono">
              {currentAngle !== null ? currentAngle.toFixed(1) : '—'}
            </span>
            {currentAngle !== null && (
              <span className="text-sm font-semibold text-accent">°</span>
            )}
          </div>
          <p className="text-[11px] text-foreground-subtle mt-1">
            {romTarget ? `Target: ${romTarget}°` : 'Real-time biomechanical angle'}
          </p>
        </div>
      </div>

      {/* 2. Repetition Counter */}
      <div className="rounded-xl border border-border bg-panel p-4 shadow-sm flex flex-col justify-between">
        <div className="flex items-center justify-between text-foreground-muted mb-2">
          <span className="text-xs font-medium uppercase tracking-wider">Repetitions</span>
          <RotateCcw className="h-4 w-4 text-emerald-400" />
        </div>
        <div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-3xl font-extrabold text-emerald-400 font-mono">
              {repCount}
            </span>
            <span className="text-xs font-semibold text-foreground-muted">Reps Completed</span>
          </div>
          <p className="text-[11px] text-foreground-subtle mt-1 capitalize">
            State: {repState ? repState.replace('_', ' ') : 'Waiting for motion'}
          </p>
        </div>
      </div>

      {/* 3. Range of Motion Progress */}
      <div className="rounded-xl border border-border bg-panel p-4 shadow-sm flex flex-col justify-between">
        <div className="flex items-center justify-between text-foreground-muted mb-2">
          <span className="text-xs font-medium uppercase tracking-wider">ROM Progress</span>
          <Target className="h-4 w-4 text-amber-400" />
        </div>
        <div>
          <div className="flex items-baseline gap-1">
            <span className="text-3xl font-extrabold text-amber-400 font-mono">
              {progressPercent !== null ? `${progressPercent}%` : '—'}
            </span>
          </div>
          {/* Progress Bar */}
          <div className="w-full bg-panel-hover rounded-full h-1.5 mt-2 overflow-hidden">
            <div
              className="bg-amber-400 h-1.5 rounded-full transition-all duration-150"
              style={{ width: `${progressPercent ?? 0}%` }}
            />
          </div>
        </div>
      </div>

      {/* 4. Calibration Status */}
      <div className="rounded-xl border border-border bg-panel p-4 shadow-sm flex flex-col justify-between">
        <div className="flex items-center justify-between text-foreground-muted mb-2">
          <span className="text-xs font-medium uppercase tracking-wider">Calibration</span>
          <Compass className="h-4 w-4 text-accent" />
        </div>
        <div>
          <div className="flex items-center gap-1.5">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                isCalibrated
                  ? 'bg-status-success'
                  : calibStatus === 'calibrating'
                  ? 'bg-amber-400 animate-ping'
                  : calibStatus === 'failed'
                  ? 'bg-status-error'
                  : 'bg-foreground-subtle'
              }`}
            />
            <span className="text-sm font-bold text-foreground capitalize">
              {calibStatus === 'calibrated'
                ? 'Baseline Active'
                : calibStatus === 'calibrating'
                ? 'Collecting...'
                : calibStatus === 'failed'
                ? 'Calibration Failed'
                : 'Uncalibrated'}
            </span>
          </div>
          <p className="text-[11px] text-foreground-subtle mt-1 truncate">
            {calibStatus === 'failed'
              ? snapshot?.calibration_error || 'Insufficient samples'
              : calibSamples > 0
              ? `${calibSamples} valid samples`
              : 'Awaiting baseline'}
          </p>
        </div>
      </div>
    </div>
  )
}
