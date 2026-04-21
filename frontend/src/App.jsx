import { useState, useEffect, useCallback, useRef } from 'react'
import { TopBar } from './components/TopBar'
import { LiveFeed } from './components/LiveFeed'
import { EventList } from './components/EventList'
import { TrendChart } from './components/TrendChart'
import { useWebSocket, api } from './hooks/useWebSocket'

const TREND_INTERVAL_MS = 60_000

export default function App() {
  const [events, setEvents] = useState([])
  const [trendData, setTrendData] = useState([])
  const [frameDataUrl, setFrameDataUrl] = useState(null)
  const [detectionCount, setDetectionCount] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [criticalAlert, setCriticalAlert] = useState(null)
  const trendTimerRef = useRef(null)

  // ── Defect rate (per hour estimate based on last hour) ────────────────────
  const defectRate = trendData.length
    ? (trendData[trendData.length - 1]?.total ?? 0)
    : 0

  // ── System status ─────────────────────────────────────────────────────────
  const status = criticalAlert ? 'HALTED' : isRunning ? 'RUNNING' : 'IDLE'

  // ── WebSocket handlers ────────────────────────────────────────────────────
  const onFrame = useCallback((msg) => {
    setIsConnected(true)
    setDetectionCount(msg.detection_count || 0)
    setFrameDataUrl(`data:image/jpeg;base64,${msg.frame}`)
  }, [])

  const onEvent = useCallback((evt) => {
    setEvents(prev => [evt, ...prev].slice(0, 50))
    if (evt.severity === 'CRITICAL') {
      setCriticalAlert(evt)
      setTimeout(() => setCriticalAlert(null), 10_000)
    }
  }, [])

  useWebSocket({ onFrame, onEvent })

  // ── Initial data load ─────────────────────────────────────────────────────
  useEffect(() => {
    api.getEvents(50).then(setEvents).catch(() => {})
    loadTrend()

    trendTimerRef.current = setInterval(loadTrend, TREND_INTERVAL_MS)
    return () => clearInterval(trendTimerRef.current)
  }, [])

  function loadTrend() {
    api.getTrend(24).then(setTrendData).catch(() => {})
  }

  // ── Pipeline controls ─────────────────────────────────────────────────────
  async function handleStart() {
    await api.startPipeline('cam0', 'line_01')
    setIsRunning(true)
    setCriticalAlert(null)
  }

  async function handleStop() {
    await api.stopPipeline()
    setIsRunning(false)
  }

  return (
    <>
      {/* Critical alert banner */}
      {criticalAlert && (
        <div className="alert-banner">
          <span>
            🚨 CRITICAL DEFECT: {criticalAlert.defect_type} on {criticalAlert.camera_id} —
            LINE HALTED
          </span>
          <button
            onClick={() => setCriticalAlert(null)}
            style={{ background: 'none', border: '1px solid rgba(255,255,255,0.4)',
                     color: '#fff', borderRadius: 3, padding: '2px 10px',
                     cursor: 'pointer', fontFamily: 'var(--mono)', fontSize: 11 }}
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="app-grid" style={{ paddingTop: criticalAlert ? 44 : 0 }}>
        <TopBar
          status={status}
          defectRate={defectRate}
          activeCamera="cam0"
          onStart={handleStart}
          onStop={handleStop}
          isRunning={isRunning}
        />
        <LiveFeed
          frameDataUrl={frameDataUrl}
          detectionCount={detectionCount}
          isConnected={isConnected}
        />
        <EventList events={events} />
        <TrendChart data={trendData} />
      </div>
    </>
  )
}
