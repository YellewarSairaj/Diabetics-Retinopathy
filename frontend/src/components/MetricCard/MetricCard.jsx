import { motion } from 'framer-motion'

export default function MetricCard({ icon: Icon, value, label, color = 'var(--primary)', delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className="glass-card flex items-center gap-4"
    >
      {/* Icon bubble */}
      <div
        className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0"
        style={{
          background: `${color}18`,
          border: `1px solid ${color}30`,
          boxShadow: `0 0 20px ${color}15`,
        }}
      >
        {Icon && <Icon className="text-2xl" style={{ color }} />}
      </div>

      {/* Data */}
      <div>
        <p className="text-2xl font-bold tabular-nums leading-tight"
          style={{ color: 'var(--text-primary)', fontFamily: 'Poppins,sans-serif' }}>
          {value ?? '—'}
        </p>
        <p className="text-sm mt-0.5" style={{ color: 'var(--text-secondary)' }}>
          {label}
        </p>
      </div>
    </motion.div>
  )
}
