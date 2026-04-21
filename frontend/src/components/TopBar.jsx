const SEV = {
  CRITICAL: { color: 'var(--danger)', bg: 'rgba(248,113,113,0.1)', border: 'rgba(248,113,113,0.3)', label: '🔴 HALT' },
  MEDIUM:   { color: 'var(--warn)',   bg: 'rgba(245,158,11,0.1)',  border: 'rgba(245,158,11,0.3)',  label: '🟡 FLAG' },
  LOW:      { color: 'var(--success)',bg: 'rgba(52,211,153,0.1)',  border: 'rgba(52,211,153,0.3)',  label: '🟢 LOG' },
}

export function TopBar({ status, defectRate, activeCamera, onStart, onStop, isRunning }) {
  return (
    <header className="topbar">
      <div className="topbar-brand">
        Industrial <span>Edge AI</span>
        <span style={{ color: 'var(--muted)', fontWeight: 400, marginLeft: 8 }}>
          · Cognizant Technoverse
        </span>
      </div>

      <div className="topbar-metrics">
        <div className="metric">
          <span className="metric-label">System</span>
          <span className="metric-value" style={{
            color: status === 'RUNNING' ? 'var(--success)' : status === 'HALTED' ? 'var(--danger)' : 'var(--muted)'
          }}>
            ● {status}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Defect Rate</span>
          <span className="metric-value">{defectRate}/hr</span>
        </div>
        <div className="metric">
          <span className="metric-label">Camera</span>
          <span className="metric-value">{activeCamera}</span>
        </div>
      </div>

      <div className="topbar-controls">
        <button className="btn btn-success" onClick={onStart} disabled={isRunning}>
          ▶ Start Pipeline
        </button>
        <button className="btn btn-danger" onClick={onStop} disabled={!isRunning}>
          ■ Stop
        </button>
      </div>
    </header>
  )
}
