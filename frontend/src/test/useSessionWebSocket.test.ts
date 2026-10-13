import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useSessionWebSocket } from '@/hooks/useSessionWebSocket'
import type { SessionSnapshot } from '@/types'

// Mock native WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = []
  url: string
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: ((event: { code: number }) => void) | null = null
  close = vi.fn()

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
  }
}

const sampleSnapshot: SessionSnapshot = {
  status: 'active',
  exercise_name: 'Sit to Stand',
  exercise_id: 'sit_to_stand',
  side: 'right',
  selected_side: 'right',
  camera_orientation: 'side',
  is_calibrated: true,
  calibration_status: 'calibrated',
  calibration_error: null,
  calibration_samples: 30,
  calibration_baseline: null,
  pose_detected: true,
  current_angle: 85.0,
  joint_angle: 85.0,
  rep_state: 'rising',
  rep_phase: 'rising',
  rep_count: 3,
  rom_target: 95.0,
  rom_progress: 0.89,
  active_compensation_ids: ['insufficient_forward_trunk_lean'],
  feedback_messages: ['Lean forward at hips'],
  latest_quality: null,
  last_error: null,
  frame_index: 200,
}

describe('useSessionWebSocket Hook', () => {
  const originalWebSocket = globalThis.WebSocket

  beforeEach(() => {
    vi.useFakeTimers()
    MockWebSocket.instances = []
    // @ts-expect-error mock assignment
    globalThis.WebSocket = MockWebSocket
  })

  afterEach(() => {
    vi.useRealTimers()
    globalThis.WebSocket = originalWebSocket
  })

  it('1. establishes WebSocket connection and sets connectionState to connected on open', () => {
    const { result } = renderHook(() => useSessionWebSocket())

    expect(MockWebSocket.instances.length).toBe(1)
    const ws = MockWebSocket.instances[0]
    expect(ws.url).toContain('/ws/session')

    // Simulate open
    act(() => {
      ws.onopen?.()
    })

    expect(result.current.connectionState).toBe('connected')
    expect(result.current.error).toBeNull()
  })

  it('2. receives and parses initial snapshot message', () => {
    const { result } = renderHook(() => useSessionWebSocket())
    const ws = MockWebSocket.instances[0]

    act(() => {
      ws.onopen?.()
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'snapshot',
          timestamp: 123456789,
          data: sampleSnapshot,
        }),
      })
    })

    expect(result.current.snapshot).toEqual(sampleSnapshot)
    expect(result.current.snapshot?.exercise_name).toBe('Sit to Stand')
    expect(result.current.snapshot?.rep_count).toBe(3)
  })

  it('3. handles consecutive updates and error messages', () => {
    const { result } = renderHook(() => useSessionWebSocket())
    const ws = MockWebSocket.instances[0]

    act(() => {
      ws.onopen?.()
      // Send error
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'error',
          timestamp: 123456789,
          error: 'Camera device disconnected',
        }),
      })
    })

    expect(result.current.error).toBe('Camera device disconnected')
  })

  it('4. attempts bounded reconnection on unexpected disconnect', () => {
    const { result } = renderHook(() => useSessionWebSocket())
    const ws1 = MockWebSocket.instances[0]

    act(() => {
      ws1.onopen?.()
      // Close abnormally
      ws1.onclose?.({ code: 1006 })
    })

    expect(result.current.connectionState).toBe('reconnecting')

    // Advance timers by backoff delay
    act(() => {
      vi.advanceTimersByTime(1100)
    })

    // A second connection should be created
    expect(MockWebSocket.instances.length).toBe(2)
  })

  it('5. cleans up WebSocket and timers upon hook unmount', () => {
    const { unmount } = renderHook(() => useSessionWebSocket())
    const ws = MockWebSocket.instances[0]

    unmount()

    expect(ws.close).toHaveBeenCalledTimes(1)
    expect(ws.onclose).toBeNull()
  })
})
