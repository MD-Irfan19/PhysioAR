import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { App } from '@/App'
import { api } from '@/lib/api'
import type { ExerciseListResponse, HealthResponse, SessionSnapshot } from '@/types'

// Mock WebSocket hook
const mockSetSnapshot = vi.fn()
const mockReconnect = vi.fn()
let mockWSState = {
  snapshot: null as SessionSnapshot | null,
  connectionState: 'connected' as const,
  error: null as string | null,
  setSnapshot: mockSetSnapshot,
  reconnect: mockReconnect,
}

vi.mock('@/hooks/useSessionWebSocket', () => ({
  useSessionWebSocket: () => mockWSState,
}))

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

describe('PhysioAR App Component', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockWSState.snapshot = null
    mockWSState.connectionState = 'connected'
    mockWSState.error = null

    vi.spyOn(api, 'getExercises').mockResolvedValue(mockExercises)
    vi.spyOn(api, 'getHealth').mockResolvedValue(mockHealth)
    vi.spyOn(api, 'startSession').mockResolvedValue({
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
      frame_index: 1,
    })
    vi.spyOn(api, 'stopSession').mockResolvedValue({
      status: 'stopped',
      message: 'Session stopped successfully.',
    })
    vi.spyOn(api, 'calibrateSession').mockResolvedValue({
      status: 'calibrating',
      message: 'Calibration requested successfully.',
    })
  })

  it('1. renders without crashing when backend is unreachable initially', async () => {
    vi.spyOn(api, 'getExercises').mockRejectedValueOnce(new Error('Network error'))
    vi.spyOn(api, 'getHealth').mockRejectedValueOnce(new Error('Network error'))

    render(<App />)

    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(/Physio/i)
    await waitFor(() => {
      expect(screen.getByText(/Network error/i)).toBeInTheDocument()
    })
  })

  it('2. loads exercises from backend API on mount', async () => {
    render(<App />)

    await waitFor(() => {
      expect(api.getExercises).toHaveBeenCalledTimes(1)
      expect(screen.getByText(/Shoulder Abduction/i)).toBeInTheDocument()
    })
  })

  it('3. does NOT start camera session automatically on page load', async () => {
    render(<App />)

    await waitFor(() => {
      expect(api.getExercises).toHaveBeenCalled()
    })

    // startSession should NOT have been called
    expect(api.startSession).not.toHaveBeenCalled()
    // Start session button should be visible
    expect(screen.getByRole('button', { name: /Start Session/i })).toBeInTheDocument()
  })

  it('4. constructs and sends valid session start request upon button click', async () => {
    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Start Session/i })).toBeInTheDocument()
    })

    // Click start session button
    fireEvent.click(screen.getByRole('button', { name: /Start Session/i }))

    await waitFor(() => {
      expect(api.startSession).toHaveBeenCalledWith({
        exercise_id: 'shoulder_abduction',
        side: 'right',
        camera_index: 0,
      })
    })
  })

  it('5. triggers stopSession and updates state cleanly when Stop Session clicked', async () => {
    mockWSState.snapshot = {
      status: 'active',
      exercise_name: 'Shoulder Abduction',
      exercise_id: 'shoulder_abduction',
      side: 'right',
      selected_side: 'right',
      camera_orientation: 'front',
      is_calibrated: true,
      calibration_status: 'calibrated',
      calibration_error: null,
      calibration_samples: 30,
      calibration_baseline: null,
      pose_detected: true,
      current_angle: 80.0,
      joint_angle: 80.0,
      rep_state: 'up',
      rep_phase: 'up',
      rep_count: 2,
      rom_target: 90.0,
      rom_progress: 0.88,
      active_compensation_ids: [],
      feedback_messages: [],
      latest_quality: null,
      last_error: null,
      frame_index: 50,
    }

    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Stop Exercise Session/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /Stop Exercise Session/i }))

    await waitFor(() => {
      expect(api.stopSession).toHaveBeenCalledTimes(1)
    })
  })

  it('6. triggers calibration request when Calibrate button clicked during active session', async () => {
    mockWSState.snapshot = {
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
      current_angle: 30.0,
      joint_angle: 30.0,
      rep_state: 'waiting',
      rep_phase: 'waiting',
      rep_count: 0,
      rom_target: 90.0,
      rom_progress: null,
      active_compensation_ids: [],
      feedback_messages: [],
      latest_quality: null,
      last_error: null,
      frame_index: 5,
    }

    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Calibrate/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /Calibrate/i }))

    await waitFor(() => {
      expect(api.calibrateSession).toHaveBeenCalledWith({
        duration_seconds: 3.0,
        min_samples: 30,
      })
    })
  })
})
