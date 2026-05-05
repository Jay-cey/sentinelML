import React, { useState } from 'react'

const API = 'http://localhost:8000'

const FEATURE_NAMES = [
  'Time',
  ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`),
  'Amount',
]

// Real confirmed-normal transaction (Class=0, row 1 of dataset)
const NORMAL_EXAMPLE = {
  Time: 406,
  V1: -1.3598071336738, V2: -0.0727811733098497, V3: 2.53634673796914,
  V4: 1.37815522427443, V5: -0.338320769942518, V6: 0.462387777762292,
  V7: 0.239598554061257, V8: 0.0986979012610507, V9: 0.363786969611213,
  V10: 0.0907941719789316, V11: -0.551599533260813, V12: -0.617800855762348,
  V13: -0.991389847235408, V14: -0.311169353699879, V15: 1.46817697209427,
  V16: -0.470400525259478, V17: 0.207971241929242, V18: 0.0257905801985591,
  V19: 0.403992960255733, V20: 0.251412098239705, V21: -0.018306777944153,
  V22: 0.277837575558899, V23: -0.110473910188767, V24: 0.0669280749146731,
  V25: 0.128539358273528, V26: -0.189114843888824, V27: 0.133558376740387,
  V28: -0.0210530534538215, Amount: 149.62,
}

// Real confirmed-fraud transaction (Class=1) that the model ACTUALLY catches
// (verified: model.predict == -1, score = -0.0597)
// The first fraud row we tried scored +0.097 — it falls in the ~65% the model misses (recall=0.356).
// This transaction is the most anomalous caught fraud in batch_01.
const FRAUD_EXAMPLE = {
  Time: 18088,
  V1: -12.2240206243564, V2: 3.85415032971366, V3: -12.4667657248211,
  V4: 9.64831072711484, V5: -2.72696132147655, V6: -4.44561038185083,
  V7: -21.9228110340575, V8: 0.320792273696859, V9: -4.43316165783721,
  V10: -11.2014000859835, V11: 9.32879925655782, V12: -13.1049334662012,
  V13: 0.88848078772943, V14: -10.140200337183, V15: 0.713465392614318,
  V16: -10.0986706554809, V17: -17.5066115481045, V18: -8.06120798966267,
  V19: 1.60687003263877, V20: -2.14718143950284, V21: -1.1598299509044,
  V22: -1.50411888189644, V23: -19.2543276173719, V24: 0.54486672601225,
  V25: -4.78160552206407, V26: -0.0077722810999307, V27: 3.05235768679424,
  V28: -0.775035651186717, Amount: 1218.89,
}

export default function PredictForm() {
  const [values, setValues] = useState(
    Object.fromEntries(FEATURE_NAMES.map(k => [k, NORMAL_EXAMPLE[k] ?? 0]))
  )
  const [activeExample, setActiveExample] = useState('normal')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  function handleChange(key, val) {
    setValues(prev => ({ ...prev, [key]: val }))
    setActiveExample(null)
  }

  function loadExample(type) {
    const src = type === 'fraud' ? FRAUD_EXAMPLE : NORMAL_EXAMPLE
    setValues(Object.fromEntries(FEATURE_NAMES.map(k => [k, src[k] ?? 0])))
    setActiveExample(type)
    setResult(null)
    setError(null)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    const features = {}
    for (const [k, v] of Object.entries(values)) {
      const parsed = parseFloat(v)
      if (isNaN(parsed)) { setError(`"${k}" must be a number.`); setLoading(false); return }
      features[k] = parsed
    }

    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="section">
      <div className="section-header">
        <span className="section-title">Transaction Predict</span>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className="btn"
            onClick={() => loadExample('normal')}
            type="button"
            style={{ borderColor: activeExample === 'normal' ? 'var(--green)' : undefined, color: activeExample === 'normal' ? 'var(--green)' : undefined }}
          >
            Load Normal
          </button>
          <button
            className="btn"
            onClick={() => loadExample('fraud')}
            type="button"
            style={{ borderColor: activeExample === 'fraud' ? 'var(--red)' : undefined, color: activeExample === 'fraud' ? 'var(--red)' : undefined }}
          >
            Load Fraud
          </button>
          <button className="btn btn-primary" form="predict-form" type="submit" disabled={loading}>
            {loading ? 'Scoring…' : '→ Score'}
          </button>
        </div>
      </div>

      <div className="section-body">
        {/* Explanation banner */}
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.68rem',
          color: 'var(--muted)',
          marginBottom: '20px',
          padding: '12px 14px',
          borderLeft: '2px solid var(--border)',
          lineHeight: 1.9,
        }}>
          <strong style={{ color: 'var(--text)' }}>Why don't large numbers trigger anomalies?</strong>
          <br />
          V1–V28 are PCA components, not raw amounts. Fraud is a specific <em>direction</em> in PCA
          space — not large magnitudes. Key fraud signals: <code style={{ color: 'var(--accent)' }}>V1 ≈ −2.3</code>,{' '}
          <code style={{ color: 'var(--accent)' }}>V9/V10/V14 strongly negative</code>,{' '}
          <code style={{ color: 'var(--accent)' }}>V4 strongly positive</code>.
          Use <strong style={{ color: 'var(--red)' }}>Load Fraud</strong> to see a real confirmed case from the training data.
        </div>

        <form id="predict-form" onSubmit={handleSubmit}>
          <div className="predict-grid">
            {FEATURE_NAMES.map(key => {
              // Highlight fields where fraud and normal differ most
              const fVal = FRAUD_EXAMPLE[key] ?? 0
              const nVal = NORMAL_EXAMPLE[key] ?? 0
              const diff = Math.abs(fVal - nVal)
              const isSignal = diff > 1.5
              return (
                <div className="field" key={key}>
                  <label htmlFor={`field-${key}`} style={{ color: isSignal ? 'var(--accent)' : undefined }}>
                    {key}{isSignal ? ' ◆' : ''}
                  </label>
                  <input
                    id={`field-${key}`}
                    type="number"
                    step="any"
                    value={values[key]}
                    onChange={e => handleChange(key, e.target.value)}
                    style={{ borderColor: isSignal && activeExample === 'fraud' ? 'rgba(200,250,100,0.3)' : undefined }}
                  />
                </div>
              )
            })}
          </div>
        </form>

        {error && (
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--red)', marginTop: '12px' }}>
            ✕ {error}
          </div>
        )}

        {result && (
          <div className="predict-result">
            <div>
              <div className="result-score" style={{ color: result.is_anomaly ? 'var(--red)' : 'var(--accent)' }}>
                {result.anomaly_score.toFixed(5)}
              </div>
              <div className="result-label">anomaly score</div>
            </div>
            <div className="result-divider" />
            <div>
              <span className={`badge ${result.is_anomaly ? 'badge-anomaly' : 'badge-normal'}`}>
                {result.is_anomaly ? 'Anomaly' : 'Normal'}
              </span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--muted)', marginTop: '6px' }}>
                {result.is_anomaly
                  ? 'Score below model threshold — flagged as fraud'
                  : 'Score above model threshold — within normal range'}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
