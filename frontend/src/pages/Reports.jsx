import { motion } from 'framer-motion'
import ReportViewer from '@/components/ReportViewer/ReportViewer'
import { usePrediction } from '@/hooks/usePrediction'

export default function Reports() {
  const { explainData, selectedFile } = usePrediction()

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold" style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
          Clinical Diagnostic Medical Report
        </h1>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          Official patient report output — ready for print, clinical audit, and PDF export
        </p>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <ReportViewer data={explainData} selectedFile={selectedFile} />
      </motion.div>
    </div>
  )
}
