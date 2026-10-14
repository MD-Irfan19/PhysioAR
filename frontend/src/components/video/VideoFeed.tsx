import React, { useState } from 'react'
import { Video, VideoOff, UserCheck, UserX, Compass, AlertCircle } from 'lucide-react'
import { getVideoStreamUrl } from '@/lib/api'
import type { SessionSnapshot, SessionStatusType } from '@/types'

interface VideoFeedProps {
  sessionStatus: SessionStatusType
  snapshot: SessionSnapshot | null
}

export const VideoFeed: React.FC<VideoFeedProps> = ({ sessionStatus, snapshot }) => {
  const [streamError, setStreamError] = useState(false)
  const isRunning =
    sessionStatus === 'active' ||
    sessionStatus === 'calibrating' ||
    sessionStatus === 'error'

  const videoUrl = isRunning ? getVideoStreamUrl() : null
  const isCalibrating =
    sessionStatus === 'calibrating' || snapshot?.calibration_status === 'calibrating'
  const isPoseDetected = snapshot?.pose_detected ?? false

  return (
    <div className="relative overflow-hidden rounded-xl border border-border bg-black shadow-md">
      {/* Top Banner / Stream Info */}
      <div className="absolute top-3 left-3 z-20 flex flex-wrap items-center gap-2">
        {/* Stream Status */}
        <div className="flex items-center gap-1.5 rounded-md bg-black/70 px-2.5 py-1 text-xs font-semibold text-white backdrop-blur-md border border-white/10">
          <span
            className={`h-2 w-2 rounded-full ${
              isRunning ? 'bg-status-success animate-pulse' : 'bg-foreground-subtle'
            }`}
          />
          <span>{isRunning ? 'LIVE FEED' : 'STREAM STANDBY'}</span>
        </div>

        {/* Pose Detection Badge */}
        {isRunning && (
          <div
            className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium backdrop-blur-md border ${
              isPoseDetected
                ? 'bg-status-success/20 text-status-success border-status-success/40'
                : 'bg-status-error/20 text-status-error border-status-error/40'
            }`}
          >
            {isPoseDetected ? (
              <>
                <UserCheck className="h-3.5 w-3.5" />
                <span>Pose Detected</span>
              </>
            ) : (
              <>
                <UserX className="h-3.5 w-3.5" />
                <span>No Pose Detected</span>
              </>
            )}
          </div>
        )}

        {/* Phase Badge */}
        {isRunning && snapshot?.rep_state && (
          <div className="rounded-md bg-accent/20 px-2.5 py-1 text-xs font-bold uppercase tracking-wider text-accent backdrop-blur-md border border-accent/40">
            Phase: {snapshot.rep_state}
          </div>
        )}
      </div>

      {/* Calibration Alert Overlay */}
      {isCalibrating && (
        <div className="absolute top-12 left-3 right-3 z-20 flex items-center justify-center gap-2 rounded-lg bg-amber-500/90 py-2 px-4 text-xs font-bold text-black shadow-lg backdrop-blur-md animate-pulse">
          <Compass className="h-4 w-4 shrink-0" />
          <span>Calibrating neutral baseline — Stand naturally with arms relaxed</span>
        </div>
      )}

      {/* Calibration Failure / Retry Guidance Overlay */}
      {!isCalibrating &&
        isRunning &&
        (snapshot?.calibration_status === 'failed' ||
          (sessionStatus === 'error' && snapshot?.calibration_error)) && (
          <div className="absolute top-12 left-3 right-3 z-20 flex items-center justify-center gap-2 rounded-lg bg-rose-500/95 py-2 px-4 text-xs font-bold text-white shadow-lg backdrop-blur-md">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>
              Calibration failed: {snapshot?.calibration_error || 'Insufficient valid frames'}. Stand in view and click "Retry Calibration".
            </span>
          </div>
        )}

      {/* Main Video Viewport */}
      <div className="relative aspect-video w-full max-h-[44vh] lg:max-h-[48vh] xl:max-h-[52vh] max-h-[460px] min-h-[220px] flex items-center justify-center bg-[#070A0F] overflow-hidden">
        {isRunning && videoUrl && !streamError ? (
          <img
            src={videoUrl}
            alt="PhysioAR Annotated Biomechanics Video Stream"
            onError={() => setStreamError(true)}
            className="h-full w-full object-contain pointer-events-none select-none"
          />
        ) : (
          <div className="flex flex-col items-center justify-center p-4 sm:p-6 text-center text-foreground-muted space-y-2.5">
            <div className="flex h-12 w-12 sm:h-14 sm:w-14 items-center justify-center rounded-2xl bg-panel text-foreground-subtle border border-border">
              {streamError ? (
                <AlertCircle className="h-6 w-6 sm:h-7 sm:w-7 text-status-error" />
              ) : (
                <VideoOff className="h-6 w-6 sm:h-7 sm:w-7 text-foreground-subtle" />
              )}
            </div>
            <div>
              <p className="text-sm font-semibold text-foreground">
                {streamError
                  ? 'Video Stream Interrupted'
                  : 'Webcam Stream Idle'}
              </p>
              <p className="text-xs text-foreground-subtle max-w-sm mt-0.5">
                {streamError
                  ? 'Unable to display camera stream. Ensure the camera is connected and backend is active.'
                  : 'Select an exercise protocol and click "Start Session" to open the video feed.'}
              </p>
            </div>
            {streamError && (
              <button
                type="button"
                onClick={() => setStreamError(false)}
                className="rounded-lg bg-panel-hover px-3 py-1.5 text-xs text-accent border border-border hover:border-accent/40"
              >
                Retry Stream Connection
              </button>
            )}
          </div>
        )}
      </div>

      {/* Bottom Sub-bar */}
      {isRunning && (
        <div className="bg-panel/90 px-4 py-2 border-t border-border flex items-center justify-between text-xs text-foreground-muted">
          <span>Exercise: <strong className="text-foreground">{snapshot?.exercise_name || 'Active Session'}</strong></span>
          <span>Tracking: <strong className="text-foreground capitalize">{snapshot?.side || 'Right'} Side</strong></span>
          <span>Frame: <strong className="text-mono text-accent">#{snapshot?.frame_index ?? 0}</strong></span>
        </div>
      )}
    </div>
  )
}
