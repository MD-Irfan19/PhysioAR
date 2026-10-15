import React from 'react'
import {
  Play,
  Square,
  Compass,
  AlertCircle,
  Camera,
  Info,
} from 'lucide-react'
import type { ExerciseItem, SessionStatusType } from '@/types'

interface SessionSetupProps {
  exercises: ExerciseItem[]
  selectedExerciseId: string
  selectedSide: 'left' | 'right'
  cameraIndex: number
  sessionStatus: SessionStatusType
  isLoading: boolean
  error: string | null
  activeExerciseId?: string | null
  activeSide?: string | null
  onSelectExercise: (id: string) => void
  onSelectSide: (side: 'left' | 'right') => void
  onChangeCameraIndex: (idx: number) => void
  onStart: () => void
  onStop: () => void
  onCalibrate: () => void
}

export const SessionSetup: React.FC<SessionSetupProps> = ({
  exercises,
  selectedExerciseId,
  selectedSide,
  cameraIndex,
  sessionStatus,
  isLoading,
  error,
  activeExerciseId,
  activeSide,
  onSelectExercise,
  onSelectSide,
  onChangeCameraIndex,
  onStart,
  onStop,
  onCalibrate,
}) => {
  const isRunning =
    sessionStatus === 'active' ||
    sessionStatus === 'calibrating' ||
    sessionStatus === 'error'
  const isCalibrating = sessionStatus === 'calibrating'
  const isError = sessionStatus === 'error'

  // When a session is active, display the exercise and side associated with that session;
  // when configuring a future session, show the pending selection.
  const displayedExerciseId =
    isRunning && activeExerciseId ? activeExerciseId : selectedExerciseId
  const displayedSide = isRunning && activeSide ? activeSide : selectedSide

  const currentExercise = exercises.find((e) => e.id === displayedExerciseId)

  // =========================================================================
  // Mode B: Active / Calibrating / Error Mode — Compact Session Toolbar
  // =========================================================================
  if (isRunning) {
    return (
      <div
        data-testid="active-session-toolbar"
        className="rounded-xl border border-border bg-panel p-3 sm:px-4 sm:py-2.5 shadow-sm space-y-2.5 w-full min-w-0"
      >
        {/* Active Error Notice (if any) */}
        {error && (
          <div
            role="alert"
            className="flex items-start gap-2 rounded-lg border border-status-error/30 bg-status-error/10 p-2 text-xs text-status-error"
          >
            <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
            <span className="leading-snug">{error}</span>
          </div>
        )}

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 min-w-0">
          {/* Left: Active Exercise Identity & Status */}
          <div className="flex items-center gap-2.5 min-w-0 flex-wrap">
            {/* Pulse Status Dot */}
            <span
              aria-hidden="true"
              className={`h-2.5 w-2.5 shrink-0 rounded-full ${
                isError
                  ? 'bg-status-error'
                  : isCalibrating
                  ? 'bg-amber-400 animate-pulse'
                  : 'bg-status-success animate-ping'
              }`}
            />

            {/* Protocol Name & Side */}
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-xs sm:text-sm font-bold text-foreground truncate">
                Protocol: {currentExercise?.name || displayedExerciseId.replace(/_/g, ' ')}
              </span>
              <span className="rounded bg-panel-subtle px-2 py-0.5 text-[11px] font-semibold text-accent border border-border uppercase tracking-wider shrink-0">
                {displayedSide} Side
              </span>
            </div>

            {/* Calibration / Session Status Tag */}
            <span
              className={`text-[11px] font-medium px-2 py-0.5 rounded border shrink-0 ${
                isError
                  ? 'border-status-error/40 bg-status-error/10 text-status-error'
                  : isCalibrating
                  ? 'border-amber-400/40 bg-amber-400/10 text-amber-400'
                  : 'border-status-success/30 bg-status-success/10 text-status-success'
              }`}
            >
              {isCalibrating
                ? 'Calibrating...'
                : isError
                ? 'Attention Required'
                : 'Live Tracking'}
            </span>

            {/* Hidden disabled input for accessibility & test contract */}
            <input
              type="text"
              aria-label="Exercise Protocol"
              disabled
              readOnly
              value={currentExercise?.name || displayedExerciseId}
              className="sr-only"
            />
          </div>

          {/* Right: Actions (Calibrate / Retry + Stop) */}
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              disabled={isLoading || isCalibrating}
              onClick={onCalibrate}
              className="flex items-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 py-1.5 px-3 text-xs font-semibold text-amber-400 hover:bg-amber-500/20 active:scale-[0.99] transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
            >
              <Compass className={`h-3.5 w-3.5 ${isCalibrating ? 'animate-spin' : ''}`} />
              <span>
                {isCalibrating
                  ? 'Calibrating Baseline...'
                  : isError
                  ? 'Retry Calibration'
                  : 'Calibrate / Reset Baseline'}
              </span>
            </button>

            <button
              type="button"
              disabled={isLoading}
              onClick={onStop}
              className="flex items-center gap-1.5 rounded-lg border border-status-error/40 bg-status-error/15 py-1.5 px-3 text-xs font-semibold text-status-error hover:bg-status-error/25 active:scale-[0.99] transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
              <span>{isLoading ? 'Stopping...' : 'Stop Exercise Session'}</span>
            </button>
          </div>
        </div>
      </div>
    )
  }

  // =========================================================================
  // Mode A: Idle / Not Started — Full Setup Card
  // =========================================================================
  return (
    <div
      data-testid="session-setup-card"
      className="rounded-xl border border-border bg-panel p-4 sm:p-5 shadow-sm space-y-4 w-full min-w-0"
    >
      <div className="flex items-center justify-between border-b border-border pb-3">
        <h2 className="text-sm sm:text-base font-semibold text-foreground flex items-center gap-2">
          <Camera className="h-4 w-4 text-accent" />
          Session Setup & Controls
        </h2>
        <span className="rounded-full bg-panel-subtle px-2.5 py-0.5 text-xs font-medium text-foreground-subtle border border-border">
          Ready
        </span>
      </div>

      {/* Error Banner */}
      {error && (
        <div
          role="alert"
          className="flex items-start gap-2.5 rounded-lg border border-status-error/30 bg-status-error/10 p-3 text-xs text-status-error"
        >
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="font-semibold">Operation Error</p>
            <p className="mt-0.5 opacity-90">{error}</p>
          </div>
        </div>
      )}

      {/* Controls Grid */}
      <div className="space-y-3">
        {/* Exercise Selection */}
        <div className="space-y-1.5">
          <label
            htmlFor="exercise-select"
            className="block text-xs font-medium text-foreground-muted uppercase tracking-wider"
          >
            Exercise Protocol
          </label>
          <select
            id="exercise-select"
            disabled={isLoading}
            value={selectedExerciseId}
            onChange={(e) => onSelectExercise(e.target.value)}
            className="w-full rounded-lg border border-border bg-panel-subtle px-3 py-2 text-sm text-foreground focus:border-accent focus:ring-1 focus:ring-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {exercises.map((ex) => (
              <option key={ex.id} value={ex.id}>
                {ex.name} ({ex.target_joint})
              </option>
            ))}
          </select>
        </div>

        {/* Side and Camera Index in a compact 2-column grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* Side Selection */}
          <div className="space-y-1.5">
            <span className="block text-xs font-medium text-foreground-muted uppercase tracking-wider">
              Target Limb / Side
            </span>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                disabled={isLoading}
                onClick={() => onSelectSide('left')}
                className={`rounded-lg py-2 text-xs font-semibold uppercase tracking-wider border transition-all ${
                  selectedSide === 'left'
                    ? 'border-accent bg-accent/15 text-accent shadow-sm'
                    : 'border-border bg-panel-subtle text-foreground-muted hover:border-panel-hover'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                Left Side
              </button>
              <button
                type="button"
                disabled={isLoading}
                onClick={() => onSelectSide('right')}
                className={`rounded-lg py-2 text-xs font-semibold uppercase tracking-wider border transition-all ${
                  selectedSide === 'right'
                    ? 'border-accent bg-accent/15 text-accent shadow-sm'
                    : 'border-border bg-panel-subtle text-foreground-muted hover:border-panel-hover'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                Right Side
              </button>
            </div>
          </div>

          {/* Camera Index Input */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center">
              <label
                htmlFor="camera-index"
                className="block text-xs font-medium text-foreground-muted uppercase tracking-wider"
              >
                Camera Index
              </label>
              <span className="text-[11px] text-foreground-subtle">0 = Default</span>
            </div>
            <input
              id="camera-index"
              type="number"
              min="0"
              max="8"
              disabled={isLoading}
              value={cameraIndex}
              onChange={(e) => onChangeCameraIndex(parseInt(e.target.value) || 0)}
              className="w-full rounded-lg border border-border bg-panel-subtle px-3 py-1.5 text-sm text-foreground focus:border-accent focus:ring-1 focus:ring-accent disabled:opacity-50 disabled:cursor-not-allowed"
            />
          </div>
        </div>
      </div>

      {/* Biomechanics Guidance Info */}
      {currentExercise && (
        <div className="rounded-lg border border-border/80 bg-panel-subtle/80 p-3 space-y-1.5 text-xs text-foreground-muted">
          <div className="flex items-center gap-1.5 font-semibold text-foreground">
            <Info className="h-3.5 w-3.5 text-accent" />
            <span>Biomechanics Guidance</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-1 text-[11px]">
            <div>
              <span className="text-foreground-subtle">Target Joint:</span>{' '}
              <span className="text-foreground capitalize">{currentExercise.target_joint}</span>
            </div>
            <div>
              <span className="text-foreground-subtle">Camera View:</span>{' '}
              <span className="text-foreground capitalize">{currentExercise.camera_orientation} View</span>
            </div>
            <div>
              <span className="text-foreground-subtle">ROM Target:</span>{' '}
              <span className="text-accent font-medium">{currentExercise.rom_target}°</span>
            </div>
          </div>
        </div>
      )}

      {/* Start Session Action */}
      <div className="pt-1">
        <button
          type="button"
          disabled={isLoading}
          onClick={onStart}
          className="w-full flex items-center justify-center gap-2 rounded-lg bg-accent py-2.5 px-4 text-sm font-semibold text-[#0B0F14] hover:bg-accent-hover active:scale-[0.99] transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-accent/20"
        >
          <Play className="h-4 w-4 fill-current" />
          <span>{isLoading ? 'Starting Session...' : 'Start Session'}</span>
        </button>
      </div>
    </div>
  )
}
