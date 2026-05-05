import React, { useEffect, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

const API = 'http://localhost:8000'

// Each "window" is a batch's score summary — we fetch drift data which
// includes PSI; we approximate a distribution by sampling the score histogram
// from the drift endpoint.  For a richer visual, we show the mean anomaly score
// per batch (returned by /drift alongside PSI).

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip">
      <div style={{ color: 'var(--muted)', marginBottom: 4 }}>Batch {label}</div>
      {payload.map(p => (
        <div key={p.name}>
          <span style={{ color: p.color }}>{p.name}: </span>
          {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  )
}

export default function ScoreChart() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch(`${API}/drift`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const windows = await res.json()

        // Baseline is batch 1 — PSI is 0 by definition
        const rows = [
          { batch: 1, psi: 0, retrain: false },
          ...windows.map(w => ({
            batch: w.batch,
            psi: w.psi,
            retrain: w.retrain_recommended,
          })),
        ]
        setData(rows)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return (
    <div className="section">
      <div className="section-header"><span className="section-title">Score Distribution</span></div>
      <div className="section-body"><p className="msg">Loading…</p></div>
    </div>
  )

  if (error) return (
    <div className="section">
      <div className="section-header"><span className="section-title">Score Distribution</span></div>
      <div className="section-body"><p className="msg error">Error: {error}</p></div>
    </div>
  )

  return (
    <div className="section">
      <div className="section-header">
        <span className="section-title">Anomaly Score Drift — Batch Over Batch</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--muted)' }}>
          PSI vs. baseline (Batch 01)
        </span>
      </div>
      <div className="section-body">
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--muted)', marginBottom: '20px' }}>
          This chart shows how the anomaly score distribution evolves across batches.
          A rising PSI signals that the model is seeing data unlike its training window.
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2e2e2e" vertical={false} />
            <XAxis
              dataKey="batch"
              tickFormatter={v => `B${v.toString().padStart(2, '0')}`}
              tick={{ fontFamily: 'IBM Plex Mono', fontSize: 11, fill: '#888' }}
              axisLine={{ stroke: '#2e2e2e' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fontFamily: 'IBM Plex Mono', fontSize: 11, fill: '#888' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={v => v.toFixed(2)}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={0.2} stroke="#ff4040" strokeDasharray="4 3" label={{ value: 'PSI 0.2', fill: '#ff4040', fontFamily: 'IBM Plex Mono', fontSize: 10 }} />
            <Line
              type="monotone"
              dataKey="psi"
              name="PSI"
              stroke="#c8fa64"
              strokeWidth={2}
              dot={{ r: 4, fill: '#c8fa64', strokeWidth: 0 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
