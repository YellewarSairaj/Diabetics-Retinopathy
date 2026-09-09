import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { HiSun, HiMoon, HiServer, HiInformationCircle } from 'react-icons/hi'
import { useTheme } from '@/hooks/useTheme'
import { apiService } from '@/services/api'
import { API_BASE_URL } from '@/utils/constants'

const FADE = (d = 0) => ({ initial: { opacity:0, y:20 }, animate: { opacity:1, y:0 }, transition: { duration:0.5, delay:d } })

export default function Settings() {
  const { theme, setDark, setLight } = useTheme()
  const [modelInfo, setModelInfo] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    apiService.getModelInfo().then(setModelInfo).catch(() => {})
    apiService.getHealth().then(setHealth).catch(() => {})
  }, [])

  return (
    <div className="flex flex-col gap-8 max-w-3xl">
      <motion.div {...FADE(0)}>
        <h1 className="text-2xl font-bold" style={{ fontFamily:'Poppins,sans-serif', color:'var(--text-primary)' }}>
          Settings &amp; Configuration
        </h1>
        <p className="text-sm mt-1" style={{ color:'var(--text-secondary)' }}>
          Customize your RetinaX AI platform experience
        </p>
      </motion.div>

      {/* Theme selector */}
      <motion.div {...FADE(0.05)} className="glass-card">
        <h3 className="font-semibold text-sm mb-4 flex items-center gap-2"
          style={{ color:'var(--text-primary)' }}>
          <HiSun style={{ color:'#f59e0b' }} /> Display Theme
        </h3>
        <div className="grid grid-cols-2 gap-3">
          {[
            { id:'dark',  Icon:HiMoon,  label:'Dark Mode',  sub:'Default · Easy on eyes', color:'#3b82f6' },
            { id:'light', Icon:HiSun,   label:'Light Mode', sub:'Bright clinical display', color:'#f59e0b' },
          ].map(({ id, Icon, label, sub, color }) => (
            <button
              key={id}
              onClick={() => id === 'dark' ? setDark() : setLight()}
              id={`theme-btn-${id}`}
              className="flex items-center gap-3 p-4 rounded-xl text-left transition-all"
              style={{
                background: theme === id ? `${color}12` : 'var(--bg-glass)',
                border: `2px solid ${theme === id ? color : 'var(--border-color)'}`,
                boxShadow: theme === id ? `0 0 16px ${color}25` : 'none',
              }}
            >
              <div className="w-9 h-9 rounded-lg flex items-center justify-center"
                style={{ background: `${color}18` }}>
                <Icon style={{ color }} />
              </div>
              <div>
                <p className="text-sm font-semibold" style={{ color:'var(--text-primary)' }}>{label}</p>
                <p className="text-xs" style={{ color:'var(--text-muted)' }}>{sub}</p>
              </div>
              {theme === id && (
                <span className="ml-auto text-xs font-bold px-2 py-0.5 rounded-full"
                  style={{ background:`${color}18`, color }}>
                  Active
                </span>
              )}
            </button>
          ))}
        </div>
      </motion.div>

      {/* API Config */}
      <motion.div {...FADE(0.1)} className="glass-card">
        <h3 className="font-semibold text-sm mb-4 flex items-center gap-2"
          style={{ color:'var(--text-primary)' }}>
          <HiServer style={{ color:'var(--primary)' }} /> API Configuration
        </h3>
        <div className="flex flex-col gap-3">
          {[
            { label:'API Base URL',   value: API_BASE_URL },
            { label:'API Version',    value: '/api/v1' },
            { label:'Docs URL',       value: `${API_BASE_URL}/docs` },
            { label:'Health Check',   value: `${API_BASE_URL}/health` },
          ].map(({ label, value }) => (
            <div key={label} className="flex items-center justify-between p-3 rounded-xl"
              style={{ background:'var(--bg-glass)', border:'1px solid var(--border-color)' }}>
              <span className="text-xs font-semibold" style={{ color:'var(--text-secondary)' }}>{label}</span>
              <code className="text-xs font-mono" style={{ color:'var(--primary)' }}>{value}</code>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Model Info */}
      <motion.div {...FADE(0.15)} className="glass-card">
        <h3 className="font-semibold text-sm mb-4 flex items-center gap-2"
          style={{ color:'var(--text-primary)' }}>
          <HiInformationCircle style={{ color:'#8b5cf6' }} /> Deployed Model Information
        </h3>
        {modelInfo ? (
          <div className="grid grid-cols-2 gap-3">
            {[
              { label:'Architecture',     value: modelInfo.architecture_name },
              { label:'Input Resolution', value: `${modelInfo.input_resolution?.[0]} × ${modelInfo.input_resolution?.[1]}` },
              { label:'Classes',          value: modelInfo.num_classes },
              { label:'XAI Layer',        value: modelInfo.target_xai_layer },
              { label:'Parameters',       value: `${(modelInfo.total_parameters/1_000_000).toFixed(2)}M` },
              { label:'Class Mapping',    value: modelInfo.class_mapping?.join(', ') || 'N/A' },
            ].map(({ label, value }) => (
              <div key={label} className="p-3 rounded-xl"
                style={{ background:'var(--bg-glass)', border:'1px solid var(--border-color)' }}>
                <p className="text-xs" style={{ color:'var(--text-muted)' }}>{label}</p>
                <p className="text-xs font-semibold mt-0.5 font-mono" style={{ color:'var(--text-primary)' }}>{value}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-center py-6" style={{ color:'var(--text-muted)' }}>
            Model info unavailable — ensure FastAPI backend is running.
          </p>
        )}
      </motion.div>

      {/* System Health */}
      <motion.div {...FADE(0.2)} className="glass-card">
        <h3 className="font-semibold text-sm mb-4 flex items-center gap-2"
          style={{ color:'var(--text-primary)' }}>
          <HiServer style={{ color: health?.model_loaded ? '#22c55e' : '#ef4444' }} />
          System Health
        </h3>
        {health ? (
          <div className="grid grid-cols-2 gap-3">
            {[
              { label:'Status',       value: health.status,            good: health.status === 'healthy' },
              { label:'Model Loaded', value: health.model_loaded ? 'Yes' : 'No', good: health.model_loaded },
              { label:'Project',      value: health.project_name,      good: true },
              { label:'Version',      value: health.version,           good: true },
            ].map(({ label, value, good }) => (
              <div key={label} className="p-3 rounded-xl"
                style={{ background: good ? 'rgba(34,197,94,0.06)' : 'rgba(239,68,68,0.06)', border:`1px solid ${good ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)'}` }}>
                <p className="text-xs" style={{ color:'var(--text-muted)' }}>{label}</p>
                <p className="text-xs font-semibold mt-0.5 font-mono capitalize"
                  style={{ color: good ? '#22c55e' : '#ef4444' }}>{String(value)}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-center py-6" style={{ color:'var(--text-muted)' }}>
            System health unavailable — ensure FastAPI backend is running.
          </p>
        )}
      </motion.div>
    </div>
  )
}
