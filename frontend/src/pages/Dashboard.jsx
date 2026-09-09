import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { HiCheckCircle, HiBadgeCheck, HiAcademicCap, HiDatabase, HiServer } from 'react-icons/hi'
import PatientHistory from '@/components/PatientHistory/PatientHistory'
import MetricCard from '@/components/MetricCard/MetricCard'
import { apiService } from '@/services/api'
import { METRICS } from '@/constants/metrics'

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)

  useEffect(() => {
    apiService.getMetrics().then(setMetrics).catch(() => {})
    apiService.getModelInfo().then(setModelInfo).catch(() => {})
  }, [])

  return (
    <div className="flex flex-col gap-8">
      {/* Page header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold" style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
          Clinical Audit &amp; Screening Dashboard
        </h1>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          Real-time system health, recent predictions history, and dataset statistics
        </p>
      </motion.div>

      {/* Stats row */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <MetricCard
          icon={HiCheckCircle}
          value={metrics?.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : METRICS.accuracy}
          label="Model Accuracy"
          color="#22c55e"
          delay={0}
        />
        <MetricCard
          icon={HiBadgeCheck}
          value={metrics?.roc_auc ? `${(metrics.roc_auc * 100).toFixed(1)}%` : METRICS.rocAuc}
          label="ROC-AUC Score"
          color="#3b82f6"
          delay={0.1}
        />
        <MetricCard
          icon={HiAcademicCap}
          value={metrics?.f1_score ? `${(metrics.f1_score * 100).toFixed(1)}%` : METRICS.f1Score}
          label="F1-Score"
          color="#8b5cf6"
          delay={0.2}
        />
        <MetricCard
          icon={HiServer}
          value={modelInfo ? `${(modelInfo.total_parameters / 1_000_000).toFixed(2)}M` : METRICS.totalParameters}
          label="Neural Parameters"
          color="#f59e0b"
          delay={0.3}
        />
      </div>

      {/* Model info card */}
      {modelInfo && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-card"
        >
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2"
            style={{ color: 'var(--text-primary)' }}>
            <HiDatabase style={{ color: 'var(--primary)' }} />
            Deployed Model Configuration
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
            {[
              { label: 'Architecture', value: modelInfo.architecture_name },
              { label: 'Input Resolution', value: `${modelInfo.input_resolution?.[0]} × ${modelInfo.input_resolution?.[1]}` },
              { label: 'Classes', value: modelInfo.num_classes },
              { label: 'XAI Layer', value: modelInfo.target_xai_layer },
            ].map(({ label, value }) => (
              <div key={label} className="rounded-xl p-3"
                style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-color)' }}>
                <p className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>{label}</p>
                <p className="text-sm font-semibold mt-1 font-mono" style={{ color: 'var(--primary)' }}>{value}</p>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Audit History */}
      <PatientHistory limit={15} />
    </div>
  )
}
