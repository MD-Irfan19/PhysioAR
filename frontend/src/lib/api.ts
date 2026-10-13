/**
 * PhysioAR Centralized API Client.
 */

import type {
  CalibrateRequest,
  ExerciseListResponse,
  GenericActionResponse,
  HealthResponse,
  SessionSnapshot,
  SessionStatusResponse,
  StartSessionRequest,
} from '@/types'

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, '') ||
  'http://localhost:8000'

export function getWebSocketUrl(): string {
  const isSecure = API_BASE_URL.startsWith('https://')
  const protocol = isSecure ? 'wss://' : 'ws://'
  const hostPath = API_BASE_URL.replace(/^https?:\/\//, '')
  return `${protocol}${hostPath}/ws/session`
}

export function getVideoStreamUrl(): string {
  return `${API_BASE_URL}/api/video/stream`
}

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${path}`
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  }

  let res: Response
  try {
    res = await fetch(url, { ...options, headers })
  } catch (err) {
    throw new ApiError(
      0,
      'Unable to connect to the PhysioAR server. Please verify the backend is running.'
    )
  }

  if (!res.ok) {
    let errorDetail = `Server returned status ${res.status}`
    try {
      const data = await res.json()
      if (typeof data.detail === 'string') {
        errorDetail = data.detail
      } else if (Array.isArray(data.detail)) {
        // Pydantic validation errors
        errorDetail = data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
      }
    } catch {
      // Body not JSON
    }
    throw new ApiError(res.status, errorDetail)
  }

  return res.json()
}

export const api = {
  getHealth: () => request<HealthResponse>('/api/health'),
  getExercises: () => request<ExerciseListResponse>('/api/exercises'),
  getSession: () => request<SessionSnapshot>('/api/session'),
  getSessionStatus: () => request<SessionStatusResponse>('/api/session/status'),
  startSession: (payload: StartSessionRequest) =>
    request<SessionSnapshot>('/api/session/start', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  stopSession: () =>
    request<GenericActionResponse>('/api/session/stop', {
      method: 'POST',
    }),
  calibrateSession: (payload: CalibrateRequest = {}) =>
    request<GenericActionResponse>('/api/session/calibrate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
}
