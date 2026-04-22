import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: 'var(--surface2)', border: '1px solid var(--border)',
                  borderRadius: 4, padding: '8px 12px', fontFamily: 'var(--mono)', fontSize: 11 }}>
      <div style={{ color: 'var(--muted)', marginBottom: 4 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {p.value}
        </div>
      ))}
    </div>
  )
}

export function TrendChart({ data }) {
  // data: [{_id: "2026-04-21T10", count: 5, critical: 1}, ...]
  const chartData = data.map(d => {
    const rawHour = d._id?.slice(11) || d._id;
    // d._id is "YYYY-MM-DDTHH", append :00:00Z to parse as UTC
    const dateObj = new Date(d._id + ":00:00Z");
    const localHour = dateObj.getHours().toString().padStart(2, '0');
    
    return {
      hour: `${localHour}:00`,
      total: d.count,
      critical: d.critical,
    };
  })

  return (
    <div className="trend-panel">
      <div className="panel-header">
        <span>24h Defect Trend</span>
        <span style={{ color: 'var(--muted)' }}>Updates every 60s</span>
      </div>
      <div style={{ flex: 1, padding: '8px 8px 4px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 4, right: 16, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="hour" tick={{ fill: 'var(--muted)', fontSize: 10, fontFamily: 'var(--mono)' }} />
            <YAxis tick={{ fill: 'var(--muted)', fontSize: 10, fontFamily: 'var(--mono)' }} allowDecimals={false} />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--muted)' }}
            />
            <Line type="monotone" dataKey="total" name="Total Defects"
                  stroke="var(--accent)" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="critical" name="Critical"
                  stroke="var(--danger)" strokeWidth={2} dot={false} strokeDasharray="4 2" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
