import React, { useEffect, useState, useCallback } from 'react'
import { Header } from '@/components/layout/Header'
import { SessionSetup } from '@/components/session/SessionSetup'
import { VideoFeed } from '@/components/video/VideoFeed'
import { MetricCards } from '@/components/metrics/MetricCards'
import { FeedbackPanel } from '@/components/feedback/FeedbackPanel'
import { QualitySummary } from '@/components/metrics/QualitySummary'
import { ErrorBoundary } from '@/components/common/ErrorBoundary'
import { useSessionWebSocket } from '@/hooks/useSessionWebSocket'
import { api, ApiError } from '@/lib/api'
import type { ExerciseItem, HealthResponse, SessionStatusType } from '@/types'

export const App: React.FC = () => {
  const [exercises, setExercises] = useState<ExerciseItem[]>([])
  const [selectedExerciseId, setSelectedExerciseId] = useState<string>('')
  const [selectedSide, setSelectedSide] = useState<'left' | 'right'>('right')
  const [cameraIndex, setCameraIndex] = useState<number>(0)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [actionError, setActionError] = useState<string | null>(null)

  const { snapshot, connectionState, error: wsError, setSnapshot, reconnect } =
    useSessionWebSocket()

  // Load exercises list, health, and any existing active session on initial mount
  useEffect(() => {
    let isMounted = true

    const loadInitialData = async () => {
      try {
        const [exercisesRes, healthRes] = await Promise.all([
          api.getExercises(),
          api.getHealth(),
        ])
        if (isMounted) {
          setExercises(exercisesRes.exercises)
          if (exercisesRes.exercises.length > 0) {
            setSelectedExerciseId(exercisesRes.exercises[0].id)
          }
          setHealth(healthRes)
          // If an active session is already ongoing on the backend, restore it immediately
          if (healthRes.runtime_running) {
            try {
              const sessionRes = await api.getSession()
              if (
                isMounted &&
                sessionRes &&
                sessionRes.status !== 'idle' &&
                sessionRes.status !== 'stopped'
              ) {
                setSnapshot(sessionRes)
                if (sessionRes.exercise_id) {
                  setSelectedExerciseId(sessionRes.exercise_id)
                }
                if (sessionRes.side === 'left' || sessionRes.side === 'right') {
                  setSelectedSide(sessionRes.side)
                }
              }
            } catch {
              // Non-blocking
            }
          }
        }
      } catch (err: any) {
        if (isMounted) {
          setActionError(err.message || 'Failed to initialize session protocols.')
        }
      }
    }

    loadInitialData()

    // Periodic health poll every 15 seconds
    const interval = setInterval(async () => {
      try {
        const h = await api.getHealth()
        if (isMounted) setHealth(h)
      } catch {
        // Silently handled in background
      }
    }, 15000)

    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [setSnapshot])

  // Start Session Action: exactly one start request followed automatically by initial calibration
  const handleStartSession = useCallback(async () => {
    if (!selectedExerciseId) {
      setActionError('Please select a valid exercise protocol.')
      return
    }

    setIsLoading(true)
    setActionError(null)

    try {
      // 1. Create exactly one session on backend
      const snap = await api.startSession({
        exercise_id: selectedExerciseId,
        side: selectedSide,
        camera_index: cameraIndex,
      })

      // Immediately mount session and indicate calibration transition
      setSnapshot({
        ...snap,
        status: 'calibrating',
        calibration_status: 'calibrating',
      })

      // Refresh health to indicate active camera hardware
      try {
        const h = await api.getHealth()
        setHealth(h)
      } catch {
        // Non-blocking
      }

      // 2. Automatically request initial calibration
      try {
        await api.calibrateSession({
          duration_seconds: 3.0,
          min_samples: 30,
        })
      } catch (calibErr: any) {
        // If initial calibration request fails, display guidance without destroying the active session
        const calibDetail =
          calibErr instanceof ApiError ? calibErr.detail : calibErr.message
        setActionError(calibDetail || 'Initial calibration baseline could not be started.')
      }
    } catch (err: any) {
      const detail = err instanceof ApiError ? err.detail : err.message
      setActionError(detail || 'Could not start session.')
    } finally {
      setIsLoading(false)
    }
  }, [selectedExerciseId, selectedSide, cameraIndex, setSnapshot])

  // Stop Session Action
  const handleStopSession = useCallback(async () => {
    setIsLoading(true)
    setActionError(null)

    try {
      await api.stopSession()
      if (snapshot) {
        setSnapshot({
          ...snapshot,
          status: 'stopped',
        })
      }
      const h = await api.getHealth()
      setHealth(h)
    } catch (err: any) {
      const detail = err instanceof ApiError ? err.detail : err.message
      setActionError(detail || 'Failed to stop exercise session.')
    } finally {
      setIsLoading(false)
    }
  }, [snapshot, setSnapshot])

  // Manual Calibrate Action (for recalibration during an active session)
  const handleCalibrate = useCallback(async () => {
    setIsLoading(true)
    setActionError(null)

    try {
      if (snapshot) {
        setSnapshot({
          ...snapshot,
          status: 'calibrating',
          calibration_status: 'calibrating',
          calibration_error: null,
        })
      }
      await api.calibrateSession({
        duration_seconds: 3.0,
        min_samples: 30,
      })
    } catch (err: any) {
      const detail = err instanceof ApiError ? err.detail : err.message
      setActionError(detail || 'Failed to request calibration baseline.')
    } finally {
      setIsLoading(false)
    }
  }, [snapshot, setSnapshot])

  const sessionStatus: SessionStatusType = snapshot?.status ?? 'idle'

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-background text-foreground flex flex-col selection:bg-accent/30 selection:text-white">
        {/* Top Application Header */}
        <Header
          connectionState={connectionState}
          health={health}
          sessionStatus={sessionStatus}
          onReconnect={reconnect}
        />

        {/* Main Workspace Container */}
        <main className="flex-1 p-3 sm:p-4 lg:p-5 max-w-[1600px] w-full mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-5 items-start">
            {/* Primary Monitoring Area (Camera Feed, Telemetry Metrics, Session Controls) */}
            <div className="contents lg:flex lg:flex-col lg:col-span-7 xl:col-span-8 lg:space-y-4 min-w-0">
              {/* Camera Feed: Mobile 1st, Desktop Left Top */}
              <div className="order-1 w-full min-w-0">
                <VideoFeed
                  sessionStatus={sessionStatus}
                  snapshot={snapshot}
                />
              </div>

              {/* Telemetry Metric Cards: Mobile 3rd, Desktop Left Below Video */}
              <div className="order-3 lg:order-2 w-full min-w-0">
                <MetricCards snapshot={snapshot} />
              </div>

              {/* Session Controls & Protocol Setup: Mobile 4th, Desktop Left Lower */}
              <div className="order-4 lg:order-3 w-full min-w-0">
                <SessionSetup
                  exercises={exercises}
                  selectedExerciseId={selectedExerciseId}
                  selectedSide={selectedSide}
                  cameraIndex={cameraIndex}
                  sessionStatus={sessionStatus}
                  isLoading={isLoading}
                  error={actionError || wsError}
                  activeExerciseId={snapshot?.exercise_id}
                  activeSide={snapshot?.side as ('left' | 'right' | null | undefined)}
                  onSelectExercise={setSelectedExerciseId}
                  onSelectSide={setSelectedSide}
                  onChangeCameraIndex={setCameraIndex}
                  onStart={handleStartSession}
                  onStop={handleStopSession}
                  onCalibrate={handleCalibrate}
                />
              </div>
            </div>

            {/* Dedicated Right-Hand Sidebar (Persistent Corrective Feedback & Repetition Quality) */}
            <div className="contents lg:flex lg:flex-col lg:col-span-5 xl:col-span-4 lg:space-y-4 min-w-0">
              {/* Real-Time Corrective Feedback: Mobile 2nd (immediately below camera), Desktop Right Top */}
              <div className="order-2 lg:order-1 w-full min-w-0">
                <FeedbackPanel snapshot={snapshot} />
              </div>

              {/* Repetition Quality Summary: Mobile 5th, Desktop Right Lower */}
              <div className="order-5 lg:order-2 w-full min-w-0">
                <QualitySummary
                  quality={snapshot?.latest_quality ?? null}
                  repNumber={snapshot?.rep_count}
                />
              </div>
            </div>
          </div>
        </main>

      {/* Restrained Footer */}
      <footer className="border-t border-border bg-panel/60 py-4 px-6 text-center text-xs text-foreground-subtle">
        PhysioAR AI-Assisted Physiotherapy Movement Analysis System — Powered by MediaPipe Pose & OpenCV
      </footer>
    </div>
  </ErrorBoundary>
  )
}

export default App
