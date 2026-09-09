import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { HiArrowLeft } from 'react-icons/hi'
import GradCAMViewer from '@/components/GradCAMViewer/GradCAMViewer'
import { usePrediction } from '@/hooks/usePrediction'

export default function GradCAM() {
  const { explainData, previewUrl } = usePrediction()
  const navigate = useNavigate()

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold" style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
          Grad-CAM Explainable AI (XAI) Visualizer
        </h1>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          Inspect deep learning activation maps and pixel-level attributions for retinal lesion localization
        </p>
      </motion.div>

      {/* No data nudge */}
      {!explainData && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-card text-center py-12 flex flex-col items-center gap-4"
        >
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            No Grad-CAM data available yet. Run a prediction first to visualize the activation maps.
          </p>
          <motion.button
            onClick={() => navigate('/prediction')}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.97 }}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold"
            style={{ background: 'linear-gradient(135deg,#2563eb,#4f46e5)', color: '#fff', boxShadow: '0 4px 16px rgba(37,99,235,0.3)' }}
          >
            <HiArrowLeft /> Go to Prediction
          </motion.button>
        </motion.div>
      )}

      {/* Viewer */}
      <GradCAMViewer data={explainData} originalUrl={previewUrl} />
    </div>
  )
}
