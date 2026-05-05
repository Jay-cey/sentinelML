import React, { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'

const API = 'http://localhost:8000'

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const { psi, status } = payload[0]?.payload || {}
  const isDrift = psi > 0.2
  return (
    <div className="custom-tooltip">
      <div style={{ color: 'var(--muted)', marginBottom: 4 }}>Batch {label}</div>
      <div>PSI: <span style={{ color: isDrift ? '#ff4040' : '#c8fa64' }}>{psi?.toFixed(4)}</span></div>
      <div>Status: <span style={{ color: isDrift ? '#ff4040' : '#3fca78' }}>{status}</span></div>
    </div>
  )
}

export default function DriftMonitor() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [retraining, setRetraining] = useState(false)
  const [retrainResult, setRetrainResult] = useState(null)

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API}/drift`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const windows = await res.json()
      setData(windows)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  async function handleRetrain() {
    setRetraining(true)
    setRetrainResult(null)
    try {
      const res = await fetch(`${API}/retrain`, { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const result = await res.json()
      setRetrainResult(result)
      await load() // Refresh drift after retrain
    } catch (e) {
      setRetrainResult({ error: e.message })
    } finally {
      setRetraining(false)
    }
  }

  const driftDetected = data.some(d => d.retrain_recommended)

  if (loading) return (
    <div className="section">
      <div className="section-header"><span className="section-title">Drift Monitor</span></div>
      <div className="section-body"><p className="msg">Loading…</p></div>
    </div>
  )

  if (error) return (
    <div className="section">
      <div className="section-header"><span className="section-title">Drift Monitor</span></div>
      <div className="section-body"><p className="msg error">Error: {error}</p></div>
    </div>
  )

  return (
    <div className="section">
      <div className="section-header">
        <span className="section-title">PSI Drift Monitor</span>
        <button
          className="btn btn-primary"
          onClick={handleRetrain}
          disabled={retraining}
        >
          {retraining ? 'Retraining…' : '↻ Retrain Model'}
        </button>
      </div>
      <div className="section-body">

        {/* Legend */}
        <div className="psi-legend">
          <div className="legend-item">
            <div className="legend-dot" style={{ background: 'var(--accent)' }} />
            Stable (PSI &lt; 0.1)
          </div>
          <div className="legend-item">
            <div className="legend-dot" style={{ background: '#f0a030' }} />
            Monitor (0.1 – 0.2)
          </div>
          <div className="legend-item">
            <div className="legend-dot" style={{ background: 'var(--red)' }} />
            Drift (PSI &gt; 0.2)
          </div>
        </div>

        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }} barSize={32}>
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
              tickFormatter={v => v.toFixed(1)}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <ReferenceLine
              y={0.2}
              stroke="#ff4040"
              strokeDasharray="4 3"
              label={{ value: '0.2 threshold', fill: '#ff4040', fontFamily: 'IBM Plex Mono', fontSize: 10, position: 'right' }}
            />
            <Bar dataKey="psi" name="PSI" radius={0}>
              {data.map((entry, i) => {
                let fill = '#c8fa64'
                if (entry.psi > 0.2) fill = '#ff4040'
                else if (entry.psi > 0.1) fill = '#f0a030'
                return <Cell key={i} fill={fill} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Status row */}
        <div className="status-row">
          {data.map(d => (
            <div key={d.batch} className="status-item">
              <div className={`status-dot ${d.retrain_recommended ? 'drift' : ''}`} />
              B{d.batch.toString().padStart(2, '0')}: {d.psi.toFixed(4)}
            </div>
          ))}
        </div>

        {/* Retrain result */}
        {retrainResult && !retrainResult.error && (
          <div style={{
            marginTop: '20px',
            padding: '14px 16px',
            border: '1px solid var(--border)',
            background: 'var(--bg)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.72rem',
            lineHeight: 1.8,
          }}>
            <div style={{ color: 'var(--accent)', marginBottom: 6 }}>✓ Retrain complete — {retrainResult.logged_at}</div>
            <div>Trained on: {retrainResult.trained_on}</div>
            <div>F1: {retrainResult.f1?.toFixed(4)} | Precision: {retrainResult.precision?.toFixed(4)} | Recall: {retrainResult.recall?.toFixed(4)}</div>
          </div>
        )}

        {retrainResult?.error && (
          <div style={{ marginTop: '16px', fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--red)' }}>
            Error: {retrainResult.error}
          </div>
        )}
      </div>
    </div>
  )
}
