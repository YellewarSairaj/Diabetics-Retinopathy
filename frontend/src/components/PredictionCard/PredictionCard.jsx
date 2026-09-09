import { motion } from 'framer-motion'
import { HiSparkles, HiExclamation } from 'react-icons/hi'
import {
  MEDICAL_DISCLAIMER,
  GRADCAM_CAPTION,
  GRADCAM_DISCLAIMER,
  getConfidenceLevel,
} from '@/utils/constants'
import { usePrediction } from '@/hooks/usePrediction'
import PredictionExplanation from '@/components/PredictionExplanation/PredictionExplanation'

function IdleState() {
  return (
    <motion.div
      key="idle"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col items-center justify-center gap-4 py-16 text-center"
    >
      <div
        className="w-20 h-20 rounded-full flex items-center justify-center"
        style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-color)' }}
      >
        <HiSparkles className="text-4xl opacity-30" style={{ color: 'var(--text-secondary)' }} />
      </div>
      <div>
        <h3
          className="font-semibold text-base"
          style={{ color: 'var(--text-primary)', fontFamily: 'Poppins,sans-serif' }}
        >
          Awaiting Patient Data
        </h3>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          Upload a fundus image and click 'Run Deep Learning &amp; XAI Diagnosis'
        </p>
      </div>
    </motion.div>
  )
}

// Default export: reads from PredictionContext automatically
export default function PredictionCard() {
  const { explainData, previewUrl } = usePrediction()
  return <PredictionCardInner data={explainData} previewUrl={previewUrl} />
}

function PredictionCardInner({ data, previewUrl }) {
  if (!data || !data.prediction) return <IdleState />

  const pred = data.prediction
  const confMeta = getConfidenceLevel(pred.confidence)

  return (
    <motion.div
      key="result"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="flex flex-col gap-6"
    >
      {/* Uploaded retinal image thumbnail header (if available) */}
      {previewUrl && (
        <div
          className="flex items-center gap-4 p-3 rounded-xl"
          style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-color)' }}
        >
          <img
            src={previewUrl}
            alt="Uploaded retinal fundus thumbnail"
            className="w-16 h-16 object-cover rounded-lg shrink-0 border"
            style={{ borderColor: 'var(--border-color)' }}
          />
          <div>
            <p className="text-xs uppercase font-semibold tracking-wider" style={{ color: 'var(--text-muted)' }}>
              Input Fundus Photograph
            </p>
            <p className="text-sm font-medium truncate max-w-xs" style={{ color: 'var(--text-primary)' }}>
              {pred.filename}
            </p>
          </div>
        </div>
      )}

      {/* Dynamic AI Prediction Summary Card Component */}
      <PredictionExplanation
        predictedClass={pred.predicted_class}
        classProbabilities={pred.class_probabilities}
        confidence={pred.confidence}
        confidenceStatus={confMeta.status}
      />

      {/* Grad-CAM Visual Overlay Preview Section */}
      {data.clinical_overlay_base64 && (
        <div
          className="flex flex-col gap-2 rounded-xl p-4"
          style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-color)' }}
        >
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
              <HiSparkles className="text-blue-500" />
              Grad-CAM Visual Heatmap Overlay
            </h4>
            <span
              className="text-xs px-2 py-0.5 rounded-full font-mono font-semibold"
              style={{
                background: 'rgba(37,99,235,0.12)',
                color: 'var(--primary)',
                border: '1px solid rgba(37,99,235,0.25)',
              }}
            >
              {data.xai_layer_used || 'top_activation'}
            </span>
          </div>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            {GRADCAM_CAPTION}
          </p>
          <div
            className="rounded-xl overflow-hidden mt-1 border"
            style={{ borderColor: 'var(--border-color)', background: '#000', maxHeight: '260px' }}
          >
            <img
              src={data.clinical_overlay_base64}
              alt="Grad-CAM Clinical Overlay"
              className="w-full h-full object-contain"
              style={{ maxHeight: '260px' }}
            />
          </div>
          <p className="text-xs italic mt-1 text-center" style={{ color: 'var(--text-muted)' }}>
            {GRADCAM_DISCLAIMER}
          </p>
        </div>
      )}

      {/* Global Medical Disclaimer Banner */}
      <div
        className="rounded-xl p-3.5 flex items-start gap-3 text-xs"
        style={{
          background: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          color: '#f59e0b',
        }}
      >
        <HiExclamation className="text-base shrink-0 mt-0.5" />
        <p className="leading-normal font-medium">
          <strong>Medical Disclaimer:</strong> {MEDICAL_DISCLAIMER}
        </p>
      </div>
    </motion.div>
  )
}

// Named export so Prediction page can pass data directly
export { PredictionCardInner }
