import { useState, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import { HiRefresh, HiClock } from 'react-icons/hi'
import { apiService } from '@/services/api'
import Loader from '@/components/Loader/Loader'

const SEVERITY_COLORS = ['#22c55e','#3b82f6','#eab308','#f97316','#ef4444']

function SeverityBadge({ name, index }) {
  const color = SEVERITY_COLORS[index ?? 0] ?? '#3b82f6'
  return (
    <span className="badge" style={{
      background: `${color}18`,
      color,
      border: `1px solid ${color}35`,
    }}>
      {name}
    </span>
  )
}

export default function PatientHistory({ limit = 15 }) {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const fetchHistory = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const data = await apiService.getPredictionHistory(limit)
      setRecords(data.records ?? [])
    } catch (e) {
      setError('Unable to load audit history. Ensure the FastAPI backend is running.')
    } finally {
      setLoading(false)
    }
  }, [limit])

  useEffect(() => { fetchHistory() }, [fetchHistory])

  return (
    <div className="glass-card flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold flex items-center gap-2"
          style={{ color: 'var(--text-primary)', fontFamily: 'Poppins,sans-serif' }}>
          <HiClock className="text-blue-400" />
          Recent Diagnostic Audit History
        </h3>
        <motion.button
          onClick={fetchHistory}
          disabled={loading}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all disabled:opacity-50"
          style={{
            background: 'var(--bg-glass)',
            border: '1px solid var(--border-color)',
            color: 'var(--text-secondary)',
          }}
        >
          <HiRefresh className={loading ? 'animate-spin' : ''} />
          Refresh
        </motion.button>
      </div>

      {/* Content */}
      {loading && <Loader text="Loading audit records..." size="sm" />}

      {error && !loading && (
        <div className="text-center py-8 text-sm" style={{ color: 'var(--text-muted)' }}>
          {error}
        </div>
      )}

      {!loading && !error && (
        <div className="overflow-x-auto">
          <table className="retina-table">
            <thead>
              <tr>
                {['Audit ID', 'Patient ID', 'Image File', 'Diagnosis', 'Confidence', 'Recommendation', 'Timestamp'].map(h => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {records.length === 0 ? (
                <tr>
                  <td colSpan={7} className="text-center py-8 text-sm"
                    style={{ color: 'var(--text-muted)' }}>
                    No clinical audit records found.
                  </td>
                </tr>
              ) : (
                records.map((r, idx) => (
                  <motion.tr
                    key={r.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.04 }}
                  >
                    <td className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
                      #{r.id}
                    </td>
                    <td className="font-semibold text-xs">{r.patient_id}</td>
                    <td className="text-xs max-w-[120px] truncate" title={r.filename}
                      style={{ color: 'var(--text-secondary)' }}>
                      {r.filename}
                    </td>
                    <td>
                      <SeverityBadge name={r.predicted_class} index={r.predicted_index} />
                    </td>
                    <td className="text-sm font-semibold tabular-nums"
                      style={{ color: 'var(--text-primary)' }}>
                      {(r.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="text-xs max-w-[160px]" style={{ color: 'var(--text-secondary)' }}>
                      {r.recommendation}
                    </td>
                    <td className="text-xs tabular-nums" style={{ color: 'var(--text-muted)' }}>
                      {r.created_at}
                    </td>
                  </motion.tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
