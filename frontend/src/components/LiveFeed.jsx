export function LiveFeed({ frameDataUrl, detectionCount, isConnected }) {
  return (
    <div className="feed-panel">
      <div className="panel-header">
        <span>Live Camera Feed</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {isConnected && <span className="live-dot" />}
          <span>{isConnected ? 'LIVE' : 'OFFLINE'}</span>
          {detectionCount > 0 && (
            <span style={{
              background: 'rgba(248,113,113,0.15)', color: 'var(--danger)',
              border: '1px solid rgba(248,113,113,0.3)', borderRadius: 3,
              padding: '1px 7px', fontSize: 9, fontFamily: 'var(--mono)',
            }}>
              {detectionCount} DETECTION{detectionCount !== 1 ? 'S' : ''}
            </span>
          )}
        </div>
      </div>

      {frameDataUrl ? (
        <img
          className="feed-img"
          src={frameDataUrl}
          alt="Live camera feed with detection overlay"
        />
      ) : (
        <div className="feed-placeholder">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
            <rect x="2" y="7" width="20" height="15" rx="2" /><path d="M12 1v6M8 4l4 3 4-3" />
          </svg>
          <span>Waiting for camera feed…</span>
          <span style={{ fontSize: 10, color: 'var(--muted)' }}>
            Start the pipeline or use /detect endpoint
          </span>
        </div>
      )}
    </div>
  )
}
