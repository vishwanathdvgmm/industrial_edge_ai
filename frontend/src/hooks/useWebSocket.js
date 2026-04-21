import { useEffect, useRef, useCallback } from 'react'

const API = 'http://localhost:8000'

export function useWebSocket({ onFrame, onEvent }) {
  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(`ws://localhost:8000/ws`)
    wsRef.current = ws

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'frame' && onFrame) onFrame(msg)
        if (msg.type === 'event' && onEvent) onEvent(msg.data)
      } catch (_) { }
    }

    ws.onclose = () => {
      // Reconnect after 2s
      reconnectTimer.current = setTimeout(connect, 2000)
    }

    ws.onerror = () => ws.close()

    // Keep-alive ping every 20s
    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping')
    }, 20000)

    ws._pingInterval = ping
  }, [onFrame, onEvent])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      if (wsRef.current) {
        clearInterval(wsRef.current._pingInterval)
        wsRef.current.close()
      }
    }
  }, [connect])

  return wsRef
}

// REST helpers
export const api = {
  startPipeline: (cameraId = 'cam0', lineId = 'line_01') =>
    fetch(`${API}/pipeline/start?camera_id=${cameraId}&line_id=${lineId}`, { method: 'POST' }).then(r => r.json()),

  stopPipeline: () =>
    fetch(`${API}/pipeline/stop`, { method: 'POST' }).then(r => r.json()),

  detectOnce: (cameraId = 'cam0') =>
    fetch(`${API}/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_id: cameraId, line_id: 'line_01' }),
    }).then(r => r.json()),

  getEvents: (limit = 50) =>
    fetch(`${API}/events?limit=${limit}`).then(r => r.json()),

  getTrend: (hours = 24) =>
    fetch(`${API}/trend?hours=${hours}`).then(r => r.json()),

  downloadPDF: (eventId) =>
    window.open(`${API}/report/${eventId}/pdf`, '_blank'),
}
