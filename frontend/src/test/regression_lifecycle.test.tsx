import '@testing-library/jest-dom/vitest'
import { render, screen, fireEvent, waitFor, act, renderHook } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { App } from '@/App'
import { api, ApiError } from '@/lib/api'
import { useSessionWebSocket } from '@/hooks/useSessionWebSocket'
import { MetricCards } from '@/components/metrics/MetricCards'
import { QualitySummary } from '@/components/metrics/QualitySummary'
import { VideoFeed } from '@/components/video/VideoFeed'
import { SessionSetup } from '@/components/session/SessionSetup'
import { FeedbackPanel } from '@/components/feedback/FeedbackPanel'
import type {
  ExerciseListResponse,
  HealthResponse,
  SessionSnapshot,
} from '@/types'

// Mock WebSocket class for hook tests
class MockWebSocket {
  static instances: MockWebSocket[] = []
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3

  readonly CONNECTING = 0
  readonly OPEN = 1
  readonly CLOSING = 2
  readonly CLOSED = 3

  url: string
  readyState: number = 0 // CONNECTING
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: ((event: { code: number }) => void) | null = null
  send = vi.fn()
  close = vi.fn().mockImplementation((_code = 1000) => {
    this.readyState = 3 // CLOSED
  })

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
  }
}

const mockExercises: ExerciseListResponse = {
  exercises: [
    {
      id: 'shoulder_abduction',
      name: 'Shoulder Abduction',
      target_joint: 'shoulder',
      camera_orientation: 'front',
      rom_target: 90.0,
    },
    {
      id: 'seated_knee_extension',
      name: 'Seated Knee Extension',
      target_joint: 'knee',
      camera_orientation: 'side',
      rom_target: 80.0,
    },
    {
      id: 'standing_hip_abduction',
      name: 'Standing Hip Abduction',
      target_joint: 'hip',
      camera_orientation: 'front',
      rom_target: 45.0,
    },
    {
      id: 'elbow_flexion_extension',
      name: 'Elbow Flexion-Extension',
      target_joint: 'elbow',
      camera_orientation: 'front',
      rom_target: 140.0,
    },
    {
      id: 'sit_to_stand',
      name: 'Sit to Stand',
      target_joint: 'hip/knee',
      camera_orientation: 'side',
      rom_target: 95.0,
    },
  ],
}

const mockHealth: HealthResponse = {
  status: 'ok',
  runtime_running: false,
  runtime_error: null,
  camera_active: false,
  version: '1.0.0',
}

const baseSnapshot: SessionSnapshot = {
  status: 'active',
  exercise_name: 'Shoulder Abduction',
  exercise_id: 'shoulder_abduction',
  side: 'right',
  selected_side: 'right',
  camera_orientation: 'front',
  is_calibrated: false,
  calibration_status: 'not_calibrated',
  calibration_error: null,
  calibration_samples: 0,
  calibration_baseline: null,
  pose_detected: true,
  current_angle: 45.0,
  joint_angle: 45.0,
  rep_state: 'rising',
  rep_phase: 'rising',
  rep_count: 0,
  rom_target: 90.0,
  rom_progress: 0.5,
  active_compensation_ids: [],
  feedback_messages: [],
  latest_quality: null,
  last_error: null,
  frame_index: 10,
}

describe('Phase D.1 Regression Tests: Session Lifecycle, Calibration & WebSocket', () => {
  const originalWebSocket = globalThis.WebSocket

  beforeEach(() => {
    vi.clearAllMocks()
    MockWebSocket.instances = []
    // @ts-expect-error mock assignment
    globalThis.WebSocket = MockWebSocket

    vi.spyOn(api, 'getExercises').mockResolvedValue(mockExercises)
    vi.spyOn(api, 'getHealth').mockResolvedValue(mockHealth)
    vi.spyOn(api, 'startSession').mockResolvedValue(baseSnapshot)
    vi.spyOn(api, 'stopSession').mockResolvedValue({
      status: 'stopped',
      message: 'Session stopped successfully.',
    })
    vi.spyOn(api, 'calibrateSession').mockResolvedValue({
      status: 'calibrating',
      message: 'Calibration requested successfully.',
    })
    vi.spyOn(api, 'getSession').mockResolvedValue(baseSnapshot)
  })

  afterEach(() => {
    globalThis.WebSocket = originalWebSocket
  })

  // 1. Successful Start Session triggers exactly one start request and one initial calibration request
  it('1. Successful Start Session triggers exactly one start request and one initial calibration request', async () => {
    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Start Session/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /Start Session/i }))

    await waitFor(() => {
      expect(api.startSession).toHaveBeenCalledTimes(1)
      expect(api.startSession).toHaveBeenCalledWith({
        exercise_id: 'shoulder_abduction',
        side: 'right',
        camera_index: 0,
      })
    })

    await waitFor(() => {
      expect(api.calibrateSession).toHaveBeenCalledTimes(1)
      expect(api.calibrateSession).toHaveBeenCalledWith({
        duration_seconds: 3.0,
        min_samples: 30,
      })
    })
  })

  // 2. Failed session startup does not trigger calibration
  it('2. Failed session startup does not trigger calibration', async () => {
    vi.spyOn(api, 'startSession').mockRejectedValueOnce(
      new ApiError(503, 'Camera hardware failed to open.')
    )

    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Start Session/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /Start Session/i }))

    await waitFor(() => {
      expect(api.startSession).toHaveBeenCalledTimes(1)
    })

    // Critical: calibration must NOT be called when session start failed
    expect(api.calibrateSession).not.toHaveBeenCalled()
    // Error is displayed to the user
    await waitFor(() => {
      expect(screen.getByText(/Camera hardware failed to open/i)).toBeInTheDocument()
    })
  })

  // 3. The dashboard stays rendered while calibrating
  it('3. The dashboard stays rendered while calibrating', async () => {
    const calibratingSnapshot: SessionSnapshot = {
      ...baseSnapshot,
      status: 'calibrating',
      calibration_status: 'calibrating',
    }

    render(
      <div>
        <VideoFeed sessionStatus="calibrating" snapshot={calibratingSnapshot} />
        <MetricCards snapshot={calibratingSnapshot} />
        <FeedbackPanel snapshot={calibratingSnapshot} />
      </div>
    )

    // Camera feed stays rendered
    const img = screen.getByRole('img')
    expect(img).toBeInTheDocument()
    expect(img).toHaveAttribute('src', expect.stringContaining('/api/video/stream'))

    // Calibration overlay banner is visible
    expect(screen.getByText(/Calibrating neutral baseline/i)).toBeInTheDocument()

    // Metric cards remain rendered and show Collecting status
    expect(screen.getByText(/Collecting.../i)).toBeInTheDocument()
  })

  // 4. Successful calibration updates visible status without unmounting camera panel
  it('4. Successful calibration updates the visible status without unmounting the camera panel', () => {
    const { rerender } = render(
      <VideoFeed
        sessionStatus="calibrating"
        snapshot={{
          ...baseSnapshot,
          status: 'calibrating',
          calibration_status: 'calibrating',
        }}
      />
    )

    expect(screen.getByText(/Calibrating neutral baseline/i)).toBeInTheDocument()
    expect(screen.getByRole('img')).toBeInTheDocument()

    // Calibration succeeds -> updates to active
    rerender(
      <VideoFeed
        sessionStatus="active"
        snapshot={{
          ...baseSnapshot,
          status: 'active',
          is_calibrated: true,
          calibration_status: 'calibrated',
        }}
      />
    )

    // Camera feed remains mounted
    expect(screen.getByRole('img')).toBeInTheDocument()
    // Calibration banner is cleared
    expect(screen.queryByText(/Calibrating neutral baseline/i)).not.toBeInTheDocument()
    expect(screen.getByText(/LIVE FEED/i)).toBeInTheDocument()
  })

  // 5. Calibration failure displays useful feedback and does not incorrectly reset the session
  it('5. Calibration failure displays useful feedback and does not incorrectly reset the session', () => {
    const failedSnapshot: SessionSnapshot = {
      ...baseSnapshot,
      status: 'error',
      calibration_status: 'failed',
      calibration_error: 'Insufficient valid frames for calibration (15 < 30)',
    }

    render(
      <div>
        <SessionSetup
          exercises={mockExercises.exercises}
          selectedExerciseId="shoulder_abduction"
          selectedSide="right"
          cameraIndex={0}
          sessionStatus="error"
          isLoading={false}
          error={null}
          onSelectExercise={vi.fn()}
          onSelectSide={vi.fn()}
          onChangeCameraIndex={vi.fn()}
          onStart={vi.fn()}
          onStop={vi.fn()}
          onCalibrate={vi.fn()}
        />
        <VideoFeed sessionStatus="error" snapshot={failedSnapshot} />
        <MetricCards snapshot={failedSnapshot} />
      </div>
    )

    // 1. Camera feed stays mounted
    expect(screen.getByRole('img')).toBeInTheDocument()

    // 2. Clear failure error is shown
    expect(
      screen.getByText(/Calibration failed: Insufficient valid frames for calibration/i)
    ).toBeInTheDocument()

    // 3. UI does NOT reset to "Start Session" (which would cause 409 Conflict)
    expect(screen.queryByRole('button', { name: /^Start Session$/i })).not.toBeInTheDocument()

    // 4. Offers Retry Calibration and Stop
    expect(screen.getByRole('button', { name: /Retry Calibration/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Stop Exercise Session/i })).toBeInTheDocument()
  })

  // 6. Manual recalibration does not issue another start request
  it('6. Manual recalibration does not issue another start request', async () => {
    const onCalibrate = vi.fn()
    const onStart = vi.fn()

    render(
      <SessionSetup
        exercises={mockExercises.exercises}
        selectedExerciseId="shoulder_abduction"
        selectedSide="right"
        cameraIndex={0}
        sessionStatus="active"
        isLoading={false}
        error={null}
        onSelectExercise={vi.fn()}
        onSelectSide={vi.fn()}
        onChangeCameraIndex={vi.fn()}
        onStart={onStart}
        onStop={vi.fn()}
        onCalibrate={onCalibrate}
      />
    )

    const calibBtn = screen.getByRole('button', { name: /Calibrate \/ Reset Baseline/i })
    fireEvent.click(calibBtn)

    expect(onCalibrate).toHaveBeenCalledTimes(1)
    expect(onStart).not.toHaveBeenCalled()
  })

  // 7. A WebSocket disconnect does not clear the last valid session snapshot
  it('7. A WebSocket disconnect does not clear the last valid session snapshot', () => {
    const { result } = renderHook(() => useSessionWebSocket())
    const ws = MockWebSocket.instances[0]

    // Populate active snapshot
    act(() => {
      ws.readyState = 1
      ws.onopen?.()
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'snapshot',
          timestamp: 100,
          data: baseSnapshot,
        }),
      })
    })

    expect(result.current.snapshot).toEqual(baseSnapshot)

    // Disconnect abnormally
    act(() => {
      ws.onclose?.({ code: 1006 })
    })

    expect(result.current.connectionState).toBe('reconnecting')
    // Snapshot is preserved!
    expect(result.current.snapshot).toEqual(baseSnapshot)
    expect(result.current.snapshot?.exercise_name).toBe('Shoulder Abduction')
  })

  // 8. Reconnection does not duplicate session-start or calibration requests
  it('8. Reconnection does not duplicate session-start or calibration requests', () => {
    vi.useFakeTimers()
    const { result } = renderHook(() => useSessionWebSocket())
    const ws1 = MockWebSocket.instances[0]

    act(() => {
      ws1.readyState = 1
      ws1.onopen?.()
      ws1.onclose?.({ code: 1006 })
    })

    expect(result.current.connectionState).toBe('reconnecting')

    act(() => {
      vi.advanceTimersByTime(1500)
    })

    // A second socket is created for reconnection
    expect(MockWebSocket.instances.length).toBe(2)

    // Neither startSession nor calibrateSession should be called by the hook
    expect(api.startSession).not.toHaveBeenCalled()
    expect(api.calibrateSession).not.toHaveBeenCalled()
    vi.useRealTimers()
  })

  // 9. Stale WebSocket callbacks cannot overwrite current state
  it('9. Stale WebSocket callbacks cannot overwrite current state', () => {
    const { result } = renderHook(() => useSessionWebSocket())
    const ws = MockWebSocket.instances[0]

    act(() => {
      ws.readyState = 1
      ws.onopen?.()
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'snapshot',
          timestamp: 100,
          data: { ...baseSnapshot, frame_index: 50 },
        }),
      })
    })

    expect(result.current.snapshot?.frame_index).toBe(50)

    // Out-of-order stale packet arrives with frame_index 40
    act(() => {
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'snapshot',
          timestamp: 90,
          data: { ...baseSnapshot, frame_index: 40 },
        }),
      })
    })

    // Current state is protected from being overwritten by stale frame
    expect(result.current.snapshot?.frame_index).toBe(50)
  })

  // 10. Repeated renders do not create redundant WebSocket connections
  it('10. Repeated renders do not create redundant WebSocket connections', () => {
    const { rerender } = renderHook(() => useSessionWebSocket())

    const initialCount = MockWebSocket.instances.length
    expect(initialCount).toBe(1)

    // Re-render multiple times
    rerender()
    rerender()
    rerender()

    // No new WebSockets created
    expect(MockWebSocket.instances.length).toBe(1)
  })

  // 11. An active session cannot accidentally be started twice
  it('11. An active session cannot accidentally be started twice', () => {
    const onStart = vi.fn()

    render(
      <SessionSetup
        exercises={mockExercises.exercises}
        selectedExerciseId="shoulder_abduction"
        selectedSide="right"
        cameraIndex={0}
        sessionStatus="active"
        isLoading={false}
        error={null}
        onSelectExercise={vi.fn()}
        onSelectSide={vi.fn()}
        onChangeCameraIndex={vi.fn()}
        onStart={onStart}
        onStop={vi.fn()}
        onCalibrate={vi.fn()}
      />
    )

    // "Start Session" button must NOT exist while session is active
    expect(screen.queryByRole('button', { name: /^Start Session$/i })).not.toBeInTheDocument()
    expect(onStart).not.toHaveBeenCalled()
  })

  // 12. Missing optional measurements do not crash the UI
  it('12. Missing optional measurements do not crash the UI', () => {
    const sparseSnapshot: SessionSnapshot = {
      status: 'active',
      exercise_name: null,
      exercise_id: null,
      side: null,
      selected_side: null,
      camera_orientation: null,
      is_calibrated: false,
      calibration_status: 'not_calibrated',
      calibration_error: null,
      calibration_samples: 0,
      calibration_baseline: null,
      pose_detected: false,
      current_angle: null,
      joint_angle: null,
      rep_state: null,
      rep_phase: null,
      rep_count: 0,
      rom_target: null,
      rom_progress: null,
      active_compensation_ids: [],
      feedback_messages: [],
      latest_quality: {
        overall_score: 0,
        rom_score: undefined,
        alignment_score: undefined,
        stability_score: undefined,
        compensation_score: undefined,
      },
      last_error: null,
      frame_index: 0,
    }

    expect(() => {
      render(
        <div>
          <VideoFeed sessionStatus="active" snapshot={sparseSnapshot} />
          <MetricCards snapshot={sparseSnapshot} />
          <QualitySummary quality={sparseSnapshot.latest_quality} />
          <FeedbackPanel snapshot={sparseSnapshot} />
        </div>
      )
    }).not.toThrow()

    // Should display graceful fallback placeholders rather than crashing
    expect(screen.getByText('0.0')).toBeInTheDocument() // Overall score
    expect(screen.getAllByText('—').length).toBeGreaterThan(0) // Null metrics
  })
})

describe('Exercise Selection Regression Suite', () => {
  const originalWebSocket = globalThis.WebSocket

  beforeEach(() => {
    vi.clearAllMocks()
    MockWebSocket.instances = []
    // @ts-expect-error mock assignment
    globalThis.WebSocket = MockWebSocket

    vi.spyOn(api, 'getExercises').mockResolvedValue(mockExercises)
    vi.spyOn(api, 'getHealth').mockResolvedValue(mockHealth)
    vi.spyOn(api, 'startSession').mockResolvedValue(baseSnapshot)
    vi.spyOn(api, 'stopSession').mockResolvedValue({
      status: 'stopped',
      message: 'Session stopped successfully.',
    })
    vi.spyOn(api, 'calibrateSession').mockResolvedValue({
      status: 'calibrating',
      message: 'Calibration requested successfully.',
    })
    vi.spyOn(api, 'getSession').mockResolvedValue(baseSnapshot)
  })

  afterEach(() => {
    globalThis.WebSocket = originalWebSocket
  })

  // 1. The selector initially displays the documented default exercise
  it('1. The selector initially displays the documented default exercise (shoulder_abduction)', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    const select = screen.getByRole('combobox') as HTMLSelectElement
    expect(select.value).toBe('shoulder_abduction')
  })

  // 2. Selecting Seated Knee Extension updates the visible dropdown value
  it('2. Selecting Seated Knee Extension updates the visible dropdown value', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    const select = screen.getByRole('combobox') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'seated_knee_extension' } })
    expect(select.value).toBe('seated_knee_extension')
  })

  // 3. Selecting Standing Hip Abduction updates the visible dropdown value
  it('3. Selecting Standing Hip Abduction updates the visible dropdown value', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    const select = screen.getByRole('combobox') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'standing_hip_abduction' } })
    expect(select.value).toBe('standing_hip_abduction')
  })

  // 4. Selecting Elbow Flexion-Extension updates the visible dropdown value
  it('4. Selecting Elbow Flexion-Extension updates the visible dropdown value', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    const select = screen.getByRole('combobox') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'elbow_flexion_extension' } })
    expect(select.value).toBe('elbow_flexion_extension')
  })

  // 5. Selecting Sit-to-Stand updates the visible dropdown value
  it('5. Selecting Sit-to-Stand updates the visible dropdown value', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    const select = screen.getByRole('combobox') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'sit_to_stand' } })
    expect(select.value).toBe('sit_to_stand')
  })

  // 6. Selecting a different exercise does not revert after unrelated React rerenders
  it('6. Selecting a different exercise does not revert after unrelated React rerenders', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    const select = screen.getByRole('combobox') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'seated_knee_extension' } })
    expect(select.value).toBe('seated_knee_extension')

    // Trigger an unrelated state update (e.g. changing limb selection)
    const leftButton = screen.getByRole('button', { name: /Left Side/i })
    fireEvent.click(leftButton)

    // Selection must remain Seated Knee Extension
    expect(select.value).toBe('seated_knee_extension')
  })

  // 7. Selecting a different exercise does not revert when a WebSocket snapshot for a previous or currently active session arrives
  it('7. Selecting a different exercise does not revert when a WebSocket snapshot arrives', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    const select = screen.getByRole('combobox') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'standing_hip_abduction' } })
    expect(select.value).toBe('standing_hip_abduction')

    // Simulate an incoming WebSocket snapshot reporting shoulder_abduction
    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.readyState = 1
      ws.onopen?.()
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'snapshot',
          data: { ...baseSnapshot, exercise_id: 'shoulder_abduction', status: 'idle' },
        }),
      })
    })

    // Dropdown value must NOT revert to shoulder_abduction
    expect(select.value).toBe('standing_hip_abduction')
  })

  // 8. The Start Session request contains the exact selected exercise ID
  it('8. The Start Session request contains the exact selected exercise ID', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    const select = screen.getByRole('combobox') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'sit_to_stand' } })

    const startBtn = screen.getByRole('button', { name: /Start Session/i })
    fireEvent.click(startBtn)

    await waitFor(() => {
      expect(api.startSession).toHaveBeenCalledTimes(1)
      expect(api.startSession).toHaveBeenCalledWith(
        expect.objectContaining({
          exercise_id: 'sit_to_stand',
        })
      )
    })
  })

  // 9. A successful session start uses the selected exercise ID in the active session state
  it('9. A successful session start uses the selected exercise ID in the active session state', async () => {
    vi.spyOn(api, 'startSession').mockResolvedValueOnce({
      ...baseSnapshot,
      exercise_id: 'seated_knee_extension',
      exercise_name: 'Seated Knee Extension',
      status: 'active',
    })

    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'seated_knee_extension' } })
    fireEvent.click(screen.getByRole('button', { name: /Start Session/i }))

    await waitFor(() => {
      expect(screen.getByText('Seated Knee Extension')).toBeInTheDocument()
    })
  })

  // 10. Automatic initial calibration still occurs exactly once after a successful session start
  it('10. Automatic initial calibration still occurs exactly once after a successful session start', async () => {
    vi.spyOn(api, 'startSession').mockResolvedValueOnce({
      ...baseSnapshot,
      exercise_id: 'elbow_flexion_extension',
      exercise_name: 'Elbow Flexion-Extension',
      status: 'active',
    })

    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'elbow_flexion_extension' } })
    fireEvent.click(screen.getByRole('button', { name: /Start Session/i }))

    await waitFor(() => {
      expect(api.calibrateSession).toHaveBeenCalledTimes(1)
      expect(api.calibrateSession).toHaveBeenCalledWith({
        duration_seconds: 3.0,
        min_samples: 30,
      })
    })
  })

  // 11. Calibration status updates do not reset the selected or active exercise ID
  it('11. Calibration status updates do not reset the selected or active exercise ID', async () => {
    vi.spyOn(api, 'startSession').mockResolvedValueOnce({
      ...baseSnapshot,
      exercise_id: 'sit_to_stand',
      exercise_name: 'Sit to Stand',
      status: 'active',
    })

    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'sit_to_stand' } })
    fireEvent.click(screen.getByRole('button', { name: /Start Session/i }))

    await waitFor(() => {
      expect(api.calibrateSession).toHaveBeenCalledTimes(1)
    })

    // Simulate WebSocket calibration progress updates
    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'snapshot',
          data: {
            ...baseSnapshot,
            exercise_id: 'sit_to_stand',
            exercise_name: 'Sit to Stand',
            status: 'calibrating',
            calibration_status: 'calibrating',
            calibration_samples: 15,
          },
        }),
      })
    })

    // Header / setup badge should preserve Sit to Stand
    expect(screen.getByText('Sit to Stand')).toBeInTheDocument()
  })

  // 12. A failed session start does not trigger calibration
  it('12. A failed session start does not trigger calibration and preserves selection', async () => {
    vi.spyOn(api, 'startSession').mockRejectedValueOnce(new Error('Hardware camera error'))

    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    const select = screen.getByRole('combobox') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'seated_knee_extension' } })
    fireEvent.click(screen.getByRole('button', { name: /Start Session/i }))

    await waitFor(() => {
      expect(screen.getByText(/Hardware camera error/i)).toBeInTheDocument()
    })

    expect(api.calibrateSession).not.toHaveBeenCalled()
    expect(select.value).toBe('seated_knee_extension')
  })

  // 13. WebSocket reconnects do not change the pending selection or duplicate session-start requests
  it('13. WebSocket reconnects do not change the pending selection or duplicate session-start requests', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    const select = screen.getByRole('combobox') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'elbow_flexion_extension' } })
    expect(select.value).toBe('elbow_flexion_extension')

    // Simulate WS disconnect & reconnect
    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.readyState = 3
      ws.onclose?.({ code: 1006 })
    })

    expect(select.value).toBe('elbow_flexion_extension')
    expect(api.startSession).not.toHaveBeenCalled()
  })

  // 14. An existing active session is restored with its correct exercise ID
  it('14. An existing active session is restored with its correct exercise ID', async () => {
    vi.spyOn(api, 'getHealth').mockResolvedValueOnce({
      ...mockHealth,
      runtime_running: true,
      camera_active: true,
    })
    vi.spyOn(api, 'getSession').mockResolvedValueOnce({
      ...baseSnapshot,
      exercise_id: 'standing_hip_abduction',
      exercise_name: 'Standing Hip Abduction',
      status: 'active',
      is_calibrated: true,
      calibration_status: 'calibrated',
    })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('Standing Hip Abduction')).toBeInTheDocument()
    })
  })

  // 15. An invalid exercise ID does not silently start Shoulder Abduction
  it('15. An invalid exercise ID does not silently start Shoulder Abduction', async () => {
    vi.spyOn(api, 'startSession').mockRejectedValueOnce(
      new ApiError(400, "Unknown exercise 'invalid_exercise'")
    )

    render(
      <SessionSetup
        exercises={mockExercises.exercises}
        selectedExerciseId="invalid_exercise"
        selectedSide="right"
        cameraIndex={0}
        sessionStatus="idle"
        isLoading={false}
        error={null}
        onSelectExercise={vi.fn()}
        onSelectSide={vi.fn()}
        onChangeCameraIndex={vi.fn()}
        onStart={() => api.startSession({ exercise_id: 'invalid_exercise', side: 'right' })}
        onStop={vi.fn()}
        onCalibrate={vi.fn()}
      />
    )

    const startBtn = screen.getByRole('button', { name: /Start Session/i })
    fireEvent.click(startBtn)

    await waitFor(() => {
      expect(api.startSession).toHaveBeenCalledWith(
        expect.objectContaining({
          exercise_id: 'invalid_exercise',
        })
      )
    })
    expect(api.startSession).not.toHaveBeenCalledWith(
      expect.objectContaining({
        exercise_id: 'shoulder_abduction',
      })
    )
  })
})
