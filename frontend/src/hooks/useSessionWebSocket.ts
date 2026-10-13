/**
 * Reusable WebSocket hook for live PhysioAR session state synchronization.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { getWebSocketUrl } from '@/lib/api'
import type { ConnectionState, SessionSnapshot, WSMessage } from '@/types'

export interface UseSessionWebSocketReturn {
  snapshot: SessionSnapshot | null
  connectionState: ConnectionState
  error: string | null
  setSnapshot: React.Dispatch<React.SetStateAction<SessionSnapshot | null>>
  reconnect: () => void
}

export function useSessionWebSocket(
  initialSnapshot: SessionSnapshot | null = null
): UseSessionWebSocketReturn {
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(initialSnapshot)
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected')
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<any>(null)
  const backoffRef = useRef<number>(1000) // Initial 1s backoff
  const isMountedRef = useRef<boolean>(true)

  const connect = useCallback((force: boolean = false) => {
    if (!isMountedRef.current) return

    // Prevent duplicate connections if already open or connecting
    if (
      !force &&
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return
    }

    // Clean up any existing connection safely
    if (wsRef.current) {
      const existingWs = wsRef.current
      existingWs.onclose = null
      existingWs.onerror = null
      existingWs.onmessage = null
      existingWs.onopen = null
      if (existingWs.readyState === WebSocket.OPEN) {
        existingWs.close(1000, 'Reconnecting')
      } else if (existingWs.readyState === WebSocket.CONNECTING) {
        existingWs.onopen = () => {
          try {
            existingWs.close(1000)
          } catch {
            // Ignored
          }
        }
      }
      wsRef.current = null
    }

    const url = getWebSocketUrl()
    let ws: WebSocket
    try {
      ws = new WebSocket(url)
    } catch (err) {
      if (isMountedRef.current) {
        setConnectionState('disconnected')
        setError('Failed to establish WebSocket connection.')
      }
      return
    }

    wsRef.current = ws

    ws.onopen = () => {
      if (!isMountedRef.current || wsRef.current !== ws) return
      setConnectionState('connected')
      setError(null)
      backoffRef.current = 1000 // Reset backoff upon successful connection
    }

    ws.onmessage = (event: MessageEvent) => {
      if (!isMountedRef.current || wsRef.current !== ws) return
      try {
        const msg: WSMessage = JSON.parse(event.data)
        if (msg && typeof msg === 'object') {
          if (msg.type === 'snapshot' && msg.data) {
            setSnapshot((prev) => {
              // Guard against stale WebSocket callbacks overwriting newer frames within the same session
              if (
                prev &&
                prev.exercise_id === msg.data.exercise_id &&
                prev.status === msg.data.status &&
                typeof prev.frame_index === 'number' &&
                typeof msg.data.frame_index === 'number' &&
                msg.data.frame_index < prev.frame_index &&
                !(prev.frame_index > 10 && msg.data.frame_index <= 5)
              ) {
                return prev
              }
              return msg.data
            })
          } else if (msg.type === 'error' && typeof msg.error === 'string') {
            setError(msg.error)
          }
        }
      } catch (e) {
        // Ignored invalid JSON
      }
    }

    ws.onerror = () => {
      if (!isMountedRef.current || wsRef.current !== ws) return
      // WebSocket standard errors do not provide details in browser
      setError('WebSocket connection error.')
    }

    ws.onclose = (event: CloseEvent) => {
      if (!isMountedRef.current || wsRef.current !== ws) return
      wsRef.current = null

      // Do not attempt reconnect on normal close
      if (event.code === 1000) {
        setConnectionState('disconnected')
        return
      }

      setConnectionState('reconnecting')

      // Exponential backoff capped at 8 seconds
      const delay = Math.min(backoffRef.current, 8000)
      backoffRef.current = delay * 1.5

      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }

      reconnectTimeoutRef.current = setTimeout(() => {
        if (isMountedRef.current) {
          connect()
        }
      }, delay)
    }
  }, [])

  const reconnect = useCallback(() => {
    backoffRef.current = 1000
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    connect(true)
  }, [connect])

  useEffect(() => {
    isMountedRef.current = true
    connect()

    return () => {
      isMountedRef.current = false
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }
      if (wsRef.current) {
        const ws = wsRef.current
        ws.onopen = null
        ws.onclose = null
        ws.onerror = null
        ws.onmessage = null
        if (ws.readyState === WebSocket.OPEN) {
          ws.close(1000, 'Clean hook unmount')
        } else if (ws.readyState === WebSocket.CONNECTING) {
          ws.onopen = () => {
            try {
              ws.close(1000)
            } catch {
              // Ignored
            }
          }
        }
        wsRef.current = null
      }
    }
  }, [connect])

  return {
    snapshot,
    connectionState,
    error,
    setSnapshot,
    reconnect,
  }
}
