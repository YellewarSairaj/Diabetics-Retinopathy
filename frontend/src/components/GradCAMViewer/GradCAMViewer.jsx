import { motion, AnimatePresence } from 'framer-motion'
import { HiSparkles, HiExclamation } from 'react-icons/hi'
import { GRADCAM_CAPTION, GRADCAM_DISCLAIMER } from '@/utils/constants'

const PANELS = [
  { key: 'original',  label: 'Original Retinal Image',       src: (d) => d._originalUrl },
  { key: 'heatmap',   label: 'Grad-CAM Thermal Heatmap',     src: (d) => d.gradcam_heatmap_base64 },
  { key: 'overlay',   label: 'Clinical Superimposed Overlay', src: (d) => d.clinical_overlay_base64 },
]

function ImagePanel({ label, src, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="flex flex-col gap-3"
    >
      <h4 className="text-sm font-semibold text-center" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </h4>
      <div
        className="rounded-2xl overflow-hidden relative"
        style={{ background: '#000', border: '1px solid var(--border-color)', aspectRatio: '1/1' }}
      >
        {src ? (
          <img
            src={src}
            alt={label}
            className="w-full h-full object-contain"
          />
        ) : (
          <div
            className="w-full h-full flex items-center justify-center p-4 text-center"
            style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}
          >
            Awaiting inference visualization
          </div>
        )}
      </div>
    </motion.div>
  )
}

export default function GradCAMViewer({ data, originalUrl }) {
  const enriched = data ? { ...data, _originalUrl: originalUrl } : null

  return (
    <div className="flex flex-col gap-6">
      {/* Task 3 Wording: Caption above image panels */}
      <div className="px-4 py-3 rounded-xl flex items-center gap-2 text-xs font-medium"
        style={{ background: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6', border: '1px solid rgba(59, 130, 246, 0.25)' }}>
        <HiSparkles className="text-base shrink-0" />
        <span>{GRADCAM_CAPTION}</span>
      </div>

      {/* Image panels */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <AnimatePresence>
          {PANELS.map(({ key, label, src }, i) => (
            <ImagePanel
              key={key}
              label={label}
              src={enriched ? src(enriched) : null}
              delay={i * 0.1}
            />
          ))}
        </AnimatePresence>
      </div>

      {/* Task 3 Wording: Under-image interpretability disclaimer */}
      <div className="px-4 py-3 rounded-xl flex items-center gap-2 text-xs"
        style={{ background: 'rgba(245, 158, 11, 0.08)', color: '#f59e0b', border: '1px solid rgba(245, 158, 11, 0.25)' }}>
        <HiExclamation className="text-base shrink-0" />
        <p className="font-medium">
          <strong>Interpretability Disclaimer:</strong> {GRADCAM_DISCLAIMER}
        </p>
      </div>

      {/* Clinical Insight Box */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-card"
      >
        <div className="flex items-center gap-2 mb-3">
          <HiSparkles className="text-lg" style={{ color: 'var(--primary)' }} />
          <h3 className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
            Clinical Visual Insight
          </h3>
          {data?.xai_layer_used && (
            <span
              className="ml-auto text-xs px-2.5 py-0.5 rounded-full font-semibold"
              style={{ background: 'rgba(37,99,235,0.12)', color: 'var(--primary)', border: '1px solid rgba(37,99,235,0.25)' }}
            >
              {data.xai_layer_used}
            </span>
          )}
        </div>
        <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          The Grad-CAM activation map extracts spatial gradients from layer{' '}
          <code
            className="px-1.5 py-0.5 rounded text-xs font-mono"
            style={{ background: 'rgba(37,99,235,0.12)', color: 'var(--primary)' }}
          >
            top_activation
          </code>
          . Warm thermal regions{' '}
          <span style={{ color: '#ef4444', fontWeight: 600 }}>(Red</span> /{' '}
          <span style={{ color: '#eab308', fontWeight: 600 }}>Yellow)</span>{' '}
          highlight focal retinal areas — such as microaneurysms, hemorrhages, or exudates — that most
          strongly contributed to the model's DR screening prediction.
          {!data && (
            <span style={{ color: 'var(--text-muted)' }}>
              {' '}Upload an image and run a prediction on the Prediction page to generate Grad-CAM visualizations.
            </span>
          )}
        </p>
      </motion.div>
    </div>
  )
}
