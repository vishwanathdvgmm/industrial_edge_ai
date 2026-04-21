import { api } from "../hooks/useWebSocket";

const SEV = {
	CRITICAL: {
		color: "var(--danger)",
		bg: "rgba(248,113,113,0.1)",
		border: "rgba(248,113,113,0.3)",
		label: "HALT LINE",
	},
	MEDIUM: {
		color: "var(--warn)",
		bg: "rgba(245,158,11,0.1)",
		border: "rgba(245,158,11,0.3)",
		label: "FLAG QC",
	},
	LOW: {
		color: "var(--success)",
		bg: "rgba(52,211,153,0.1)",
		border: "rgba(52,211,153,0.3)",
		label: "LOG ONLY",
	},
};

function EventCard({ event }) {
	const s = SEV[event.severity] || SEV.LOW;

	return (
		<div
			className="event-card"
			style={{ borderLeftColor: s.color }}
			title="Click to download PDF report"
			onClick={() =>
				event.pdf_gridfs_id && api.downloadPDF(event.event_id)
			}
		>
			<div className="event-card-top">
				<span className="event-type">{event.defect_type}</span>
				<span
					className="event-badge"
					style={{
						color: s.color,
						background: s.bg,
						borderColor: s.border,
					}}
				>
					{s.label}
				</span>
			</div>
			<div className="event-cause">{event.cause_hypothesis}</div>
			<div className="event-meta">
				conf {((event.confidence || 0) * 100).toFixed(0)}% ·{" "}
				{event.zone} · {event.camera_id} ·{" "}
				{new Date(event.timestamp + (event.timestamp?.endsWith('Z') ? '' : 'Z')).toLocaleTimeString()}
			</div>
		</div>
	);
}

export function EventList({ events }) {
	return (
		<div className="events-panel">
			<div className="panel-header">
				<span>Defect Events</span>
				<span style={{ color: "var(--accent)" }}>
					{events.length} total
				</span>
			</div>
			<div className="events-list">
				{events.length === 0 && (
					<div
						style={{
							color: "var(--muted)",
							fontFamily: "var(--mono)",
							fontSize: 11,
							padding: 16,
							textAlign: "center",
						}}
					>
						No defect events yet
					</div>
				)}
				{events.map((e, i) => (
					<EventCard key={e.event_id || i} event={e} />
				))}
			</div>
		</div>
	);
}
