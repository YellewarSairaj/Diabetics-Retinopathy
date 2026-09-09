import { motion } from 'framer-motion'
import { HiChip } from 'react-icons/hi'
import UploadBox from '@/components/UploadBox/UploadBox'
import { PredictionCardInner } from '@/components/PredictionCard/PredictionCard'
import ProbabilityChart from '@/components/ProbabilityChart/ProbabilityChart'
import Loader from '@/components/Loader/Loader'
import { usePrediction } from '@/hooks/usePrediction'

export default function Prediction() {
  const { explainData, isLoading, error, runAnalysis, selectedFile } = usePrediction()

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold" style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
          Diabetic Retinopathy Automated Screening
        </h1>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          Upload a retinal fundus photograph for immediate AI inference and severity classification
        </p>
      </motion.div>

      {/* Main layout: upload + result side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Upload Box */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-card flex flex-col gap-6"
        >
          <UploadBox />

          {/* Analyze Button */}
          <motion.button
            onClick={() => runAnalysis('gradcam')}
            disabled={!selectedFile || isLoading}
            whileHover={selectedFile && !isLoading ? { scale: 1.02, boxShadow: '0 8px 30px rgba(37,99,235,0.4)' } : {}}
            whileTap={selectedFile && !isLoading ? { scale: 0.98 } : {}}
            className="w-full flex items-center justify-center gap-2 py-3.5 rounded-xl text-sm font-semibold transition-all"
            style={{
              background: selectedFile && !isLoading
                ? 'linear-gradient(135deg,#2563eb,#4f46e5)'
                : 'var(--bg-glass)',
              color: selectedFile && !isLoading ? '#fff' : 'var(--text-muted)',
              border: '1px solid var(--border-color)',
              cursor: selectedFile && !isLoading ? 'pointer' : 'not-allowed',
              boxShadow: selectedFile && !isLoading ? '0 4px 16px rgba(37,99,235,0.3)' : 'none',
            }}
          >
            {isLoading ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  className="w-4 h-4 rounded-full border-2 border-t-transparent"
                  style={{ borderColor: 'rgba(255,255,255,0.4)', borderTopColor: '#fff' }}
                />
                Analyzing Retinal Lesions &amp; Generating Heatmaps...
              </>
            ) : (
              <>
                <HiChip className="text-base" />
                Run Deep Learning &amp; XAI Diagnosis
              </>
            )}
          </motion.button>

          {/* API error */}
          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="px-4 py-3 rounded-xl text-sm"
              style={{ background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.25)' }}
            >
              ⚠ {error}
            </motion.div>
          )}
        </motion.div>

        {/* Prediction Result Card */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.15 }}
          className="glass-card"
        >
          {isLoading ? (
            <Loader text="Analyzing Retinal Lesions &amp; Generating Heatmaps..." size="md" />
          ) : (
            <PredictionCardInner data={explainData} />
          )}
        </motion.div>
      </div>

      {/* Chart below — only when results exist */}
      {explainData?.prediction && !isLoading && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-card"
        >
          <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Severity Grade Probability Distribution Chart
          </h3>
          <ProbabilityChart
            probabilities={explainData.prediction.class_probabilities}
            predictedIndex={explainData.prediction.predicted_index}
          />
        </motion.div>
      )}
    </div>
  )
}
