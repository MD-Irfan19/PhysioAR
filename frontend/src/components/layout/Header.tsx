import React from 'react'
import { Activity, Wifi, WifiOff, RefreshCw, Video, VideoOff } from 'lucide-react'
import type { ConnectionState, HealthResponse, SessionStatusType } from '@/types'

interface HeaderProps {
  connectionState: ConnectionState
  health: HealthResponse | null
  sessionStatus?: SessionStatusType
  onReconnect?: () => void
}

export const Header: React.FC<HeaderProps> = ({
  connectionState,
  health,
  sessionStatus = 'idle',
  onReconnect,
}) => {
  const isConnected = connectionState === 'connected'
  const isReconnecting = connectionState === 'reconnecting'
  const isCameraActive = health?.camera_active ?? false

  return (
    <header className="border-b border-border bg-panel px-6 py-4 flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/15 text-accent border border-accent/30 shadow-inner">
          <Activity className="h-5 w-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight text-foreground">
              Physio<span className="text-accent">AR</span>
            </h1>
            <span className="rounded bg-panel-hover px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-accent border border-accent/20">
              Clinical Biomechanics
            </span>
          </div>
          <p className="text-xs text-foreground-muted">
            Computer-Vision Augmented Physical Therapy Assessment
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Camera hardware status */}
        <div
          className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium border ${
            isCameraActive
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              : 'bg-panel-hover text-foreground-subtle border-border'
          }`}
          title={isCameraActive ? 'Webcam active' : 'Webcam idle'}
        >
          {isCameraActive ? (
            <Video className="h-3.5 w-3.5 text-emerald-400" />
          ) : (
            <VideoOff className="h-3.5 w-3.5" />
          )}
          <span>{isCameraActive ? 'Camera Live' : 'Camera Standby'}</span>
        </div>

        {/* WebSocket Connection status */}
        <div
          className={`flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium border transition-colors ${
            isConnected
              ? 'bg-status-success/10 text-status-success border-status-success/20'
              : isReconnecting
              ? 'bg-status-warning/10 text-status-warning border-status-warning/20'
              : 'bg-status-error/10 text-status-error border-status-error/20'
          }`}
        >
          {isConnected ? (
            <Wifi className="h-3.5 w-3.5" />
          ) : isReconnecting ? (
            <RefreshCw className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <WifiOff className="h-3.5 w-3.5" />
          )}
          <span className="capitalize">{connectionState}</span>
          {!isConnected && onReconnect && (
            <button
              onClick={onReconnect}
              type="button"
              className="ml-1 underline hover:opacity-80 text-[11px]"
            >
              Retry
            </button>
          )}
        </div>

        {/* Session Status Pill */}
        <div
          className={`rounded-full px-3 py-1 text-xs font-medium border uppercase tracking-wider ${
            sessionStatus === 'active'
              ? 'bg-accent/15 text-accent border-accent/30 animate-pulseSlow'
              : sessionStatus === 'calibrating'
              ? 'bg-amber-500/15 text-amber-400 border-amber-500/30'
              : sessionStatus === 'error'
              ? 'bg-status-error/15 text-status-error border-status-error/30'
              : 'bg-panel-hover text-foreground-muted border-border'
          }`}
        >
          {sessionStatus}
        </div>
      </div>
    </header>
  )
}
