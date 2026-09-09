import { useState } from 'react'
import { motion } from 'framer-motion'
import { HiDocumentDownload, HiRefresh } from 'react-icons/hi'
import { RiEyeLine } from 'react-icons/ri'
import { SEVERITY_COLORS, getDynamicClinicalSupport } from '@/utils/constants'
import { apiService } from '@/services/api'

export default function ReportViewer({ data, selectedFile }) {
  const [downloading, setDownloading] = useState(false)
  const [dlError, setDlError] = useState('')

  const pred = data?.prediction

  const downloadPdf = async () => {
    if (!selectedFile) {
      setDlError('Please upload and analyze a fundus image first.')
      return
    }
    setDownloading(true)
    setDlError('')
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      const blob = await apiService.downloadPdfReport(formData)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `RetinaX_Report_${selectedFile.name}_${Date.now()}.pdf`
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (e) {
      setDlError('Report generation failed. Please try downloading again.')
    } finally {
      setDownloading(false)
    }
  }

  const color = pred ? (SEVERITY_COLORS[pred.predicted_index] ?? 'var(--primary)') : 'var(--primary)'

  return (
    <div className="report-paper">
      {/* Report Header */}
      <div className="flex items-start justify-between mb-8 pb-6"
        style={{ borderBottom: '1px solid var(--border-color)' }}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg,#2563eb,#7c3aed)' }}>
            <RiEyeLine className="text-white text-lg" />
          </div>
          <div>
            <h2 className="text-lg font-bold" style={{ color: 'var(--primary)', fontFamily: 'Poppins,sans-serif' }}>
              RetinaX AI Diagnostic Report
            </h2>
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              Ophthalmic Explainable AI Decision Support System
            </p>
          </div>
        </div>

        <motion.button
          onClick={downloadPdf}
          disabled={downloading || !selectedFile}
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.97 }}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          style={{
            background: 'linear-gradient(135deg,#2563eb,#4f46e5)',
            color: '#fff',
            boxShadow: '0 4px 16px rgba(37,99,235,0.35)',
          }}
        >
          {downloading
            ? <><HiRefresh className="animate-spin" /> Generating...</>
            : <><HiDocumentDownload className="text-base" /> Download PDF Report</>
          }
        </motion.button>
      </div>

      {/* Patient Info Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        {[
          { label: 'Patient ID', value: pred ? `PAT-${selectedFile?.name?.substring(0, 8) || '8823901'}` : 'PAT-8823901' },
          { label: 'Examination Date', value: new Date().toISOString().split('T')[0] },
          { label: 'Primary AI Impression', value: pred?.predicted_class ?? 'Pending Analysis', highlight: true },
        ].map(({ label, value, highlight }) => (
          <div key={label} className="rounded-xl p-4"
            style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-color)' }}>
            <label className="text-xs uppercase tracking-wider font-semibold"
              style={{ color: 'var(--text-muted)' }}>
              {label}
            </label>
            <p className="text-sm font-semibold mt-1"
              style={{ color: highlight && pred ? color : 'var(--text-primary)' }}>
              {value}
            </p>
          </div>
        ))}
      </div>

      {/* Diagnostic Summary */}
      {pred && (
        <div className="mb-8">
          <h4 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
            Diagnostic Summary &amp; Recommendation
          </h4>
          <p className="text-sm leading-relaxed font-medium" style={{ color: 'var(--text-primary)' }}>
            {getDynamicClinicalSupport(pred.predicted_class, pred.confidence)}
          </p>
          <div className="mt-3 flex items-center gap-3">
            <div className="text-xs font-semibold px-3 py-1 rounded-full"
              style={{ background: `${color}18`, color, border: `1px solid ${color}35` }}>
              Model Confidence: {(pred.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {/* Overlay Image */}
      {data?.clinical_overlay_base64 && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
            Visual Explanation Overlay (Grad-CAM)
          </h4>
          <img
            src={data.clinical_overlay_base64}
            alt="Clinical Grad-CAM overlay"
            className="w-full rounded-xl object-contain"
            style={{ maxHeight: '300px', background: '#000', border: '1px solid var(--border-color)' }}
          />
        </div>
      )}

      {/* No data placeholder */}
      {!pred && (
        <div className="text-center py-10">
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Run a prediction first to generate the clinical report.
          </p>
        </div>
      )}

      {/* Download error */}
      {dlError && (
        <p className="mt-4 text-sm px-3 py-2 rounded-lg"
          style={{ background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.25)' }}>
          {dlError}
        </p>
      )}
    </div>
  )
}
