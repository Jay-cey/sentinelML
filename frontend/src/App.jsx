import React, { useState } from 'react'
import ScoreChart from './components/ScoreChart'
import DriftMonitor from './components/DriftMonitor'
import PredictForm from './components/PredictForm'

export default function App() {
  const [activeTab, setActiveTab] = useState('predict')

  const tabs = [
    { id: 'predict', label: 'Predict' },
    { id: 'drift', label: 'Drift Monitor' },
    { id: 'scores', label: 'Score Distribution' },
  ]

  return (
    <div className="app">
      <header className="app-header">
        <h1>sentinel_ml</h1>
        <span className="tag">Fraud Monitor</span>
      </header>

      {/* Tab nav */}
      <div style={{ display: 'flex', gap: '2px', marginBottom: '24px', borderBottom: '1px solid var(--border)' }}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.72rem',
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              padding: '10px 18px',
              border: 'none',
              background: activeTab === tab.id ? 'var(--surface)' : 'transparent',
              color: activeTab === tab.id ? 'var(--text)' : 'var(--muted)',
              cursor: 'pointer',
              borderBottom: activeTab === tab.id ? '2px solid var(--accent)' : '2px solid transparent',
              marginBottom: '-1px',
              transition: 'color 0.15s',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="sections">
        {activeTab === 'predict' && <PredictForm />}
        {activeTab === 'drift' && <DriftMonitor />}
        {activeTab === 'scores' && <ScoreChart />}
      </div>
    </div>
  )
}
