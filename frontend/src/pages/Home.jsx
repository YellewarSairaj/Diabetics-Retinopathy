import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  HiUpload, HiChartBar, HiCheckCircle, HiBadgeCheck,
  HiAcademicCap, HiDatabase, HiShieldCheck, HiExclamationCircle,
  HiExclamation, HiXCircle,
} from 'react-icons/hi'
import { RiEyeLine } from 'react-icons/ri'
import MetricCard from '@/components/MetricCard/MetricCard'
import { apiService } from '@/services/api'

const FADE_UP = { initial: { opacity: 0, y: 30 }, animate: { opacity: 1, y: 0 } }

export default function Home() {
  const navigate = useNavigate()
  const [metrics, setMetrics] = useState({
    accuracy: '84.20%', auc: '94.50%', f1: '83.80%', params: '4.06M',
  })

  useEffect(() => {
    apiService.getMetrics().then(d => {
      if (d) {
        setMetrics({
          accuracy: d.accuracy ? `${(d.accuracy * 100).toFixed(1)}%` : '84.20%',
          auc: d.roc_auc ? `${(d.roc_auc * 100).toFixed(1)}%` : '94.50%',
          f1: d.f1_score ? `${(d.f1_score * 100).toFixed(1)}%` : '83.80%',
          params: '4.06M',
        })
      }
    }).catch(() => {})
  }, [])

  return (
    <div className="flex flex-col gap-16">

      {/* ── Hero Section ── */}
      <section className="flex flex-col lg:flex-row items-center gap-12 pt-4">
        {/* Left: Text */}
        <motion.div {...FADE_UP} transition={{ duration: 0.6 }} className="flex-1 flex flex-col gap-6">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full w-fit text-xs font-semibold"
            style={{ background: 'rgba(37,99,235,0.12)', color: 'var(--primary)', border: '1px solid rgba(37,99,235,0.25)' }}>
            <RiEyeLine />
            EfficientNet-B0 · Grad-CAM · APTOS 2019
          </div>

          <h1 className="text-4xl lg:text-5xl font-bold leading-tight"
            style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
            Next-Gen Diabetic<br />
            Retinopathy{' '}
            <span className="gradient-text">Screening &amp; Visual XAI</span>
          </h1>

          <p className="text-base leading-relaxed max-w-xl" style={{ color: 'var(--text-secondary)' }}>
            Empowering clinicians with automated deep learning risk stratification, Ben Graham fundus
            image enhancement, and instant Grad-CAM visual lesion localization.
          </p>

          <div className="flex flex-wrap gap-3">
            <motion.button
              onClick={() => navigate('/prediction')}
              whileHover={{ scale: 1.04, boxShadow: '0 8px 30px rgba(37,99,235,0.45)' }}
              whileTap={{ scale: 0.97 }}
              className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold"
              style={{ background: 'linear-gradient(135deg,#2563eb,#4f46e5)', color: '#fff', boxShadow: '0 4px 16px rgba(37,99,235,0.35)' }}
            >
              <HiUpload /> Upload Fundus Image
            </motion.button>
            <motion.button
              onClick={() => navigate('/dashboard')}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold"
              style={{ background: 'var(--bg-glass)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
            >
              <HiChartBar /> View Dashboard Metrics
            </motion.button>
          </div>
        </motion.div>

        {/* Right: Hero card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="glass-card flex flex-col gap-4 w-full lg:w-auto lg:min-w-[320px]"
        >
          <img
            src={`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/outputs/gradcam_sample_overlay.png`}
            alt="Retinal Grad-CAM overlay"
            className="w-full rounded-xl object-contain"
            style={{ maxHeight: '220px', background: '#000' }}
            onError={e => {
              e.target.src = 'https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&w=600&q=80'
            }}
          />
          <div className="flex items-center justify-between">
            <div>
              <p className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>EfficientNet-B0</p>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Grad-CAM Lesion Heatmap Overlay</p>
            </div>
            <span className="text-xs font-bold px-3 py-1 rounded-full"
              style={{ background: 'rgba(37,99,235,0.15)', color: 'var(--primary)', border: '1px solid rgba(37,99,235,0.3)' }}>
              94.5% AUC
            </span>
          </div>
        </motion.div>
      </section>

      {/* ── Metrics Grid ── */}
      <section>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard icon={HiCheckCircle} value={metrics.accuracy} label="Model Accuracy" color="#22c55e" delay={0} />
          <MetricCard icon={HiBadgeCheck}  value={metrics.auc}      label="Macro OvR ROC-AUC" color="#3b82f6" delay={0.1} />
          <MetricCard icon={HiAcademicCap} value={metrics.f1}       label="Weighted F1-Score" color="#8b5cf6" delay={0.2} />
          <MetricCard icon={HiDatabase}    value={metrics.params}   label="Neural Parameters" color="#f59e0b" delay={0.3} />
        </div>
      </section>

      {/* ── Overview & Clinical Protocol ── */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Project Overview */}
        <motion.div
          {...FADE_UP} transition={{ duration: 0.5, delay: 0.1 }}
          className="glass-card flex flex-col gap-4"
        >
          <div>
            <h2 className="font-bold text-lg flex items-center gap-2"
              style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
              <HiDatabase style={{ color: 'var(--primary)' }} />
              Project Overview
            </h2>
            <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
              Enterprise Ophthalmic AI Platform Specification
            </p>
          </div>
          <ul className="flex flex-col gap-3">
            {[
              ['APTOS 2019 Dataset', 'Trained on 3,662 high-resolution retinal fundus photographs.'],
              ['Ben Graham Preprocessing', 'Local color subtraction & circular border cropping for lesion sharpening.'],
              ['Grad-CAM & Grad-CAM++', 'Visual explainability maps showing exact retinal pixel attributions.'],
              ['Class Weighting', 'Compensates for severe 9.35:1 class imbalance between No DR and Severe DR.'],
            ].map(([title, desc]) => (
              <li key={title} className="flex items-start gap-3 text-sm">
                <HiCheckCircle className="text-green-500 shrink-0 mt-0.5" />
                <span style={{ color: 'var(--text-secondary)' }}>
                  <strong style={{ color: 'var(--text-primary)' }}>{title}: </strong>{desc}
                </span>
              </li>
            ))}
          </ul>
        </motion.div>

        {/* Clinical Protocol */}
        <motion.div
          {...FADE_UP} transition={{ duration: 0.5, delay: 0.2 }}
          className="glass-card flex flex-col gap-4"
        >
          <div>
            <h2 className="font-bold text-lg flex items-center gap-2"
              style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
              <HiExclamation style={{ color: '#f59e0b' }} />
              Clinical Recommendation Protocol
            </h2>
            <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
              Actionable Guidelines per DR Severity Grade
            </p>
          </div>
          <ul className="flex flex-col gap-3">
            {[
              { icon: HiShieldCheck,       color: '#22c55e', label: 'Grade 0 (No DR)',              desc: 'Annual routine ophthalmic screening.' },
              { icon: HiExclamationCircle, color: '#3b82f6', label: 'Grade 1 (Mild)',               desc: 'Microaneurysms detected. Follow-up in 6–12 months.' },
              { icon: HiExclamation,       color: '#eab308', label: 'Grade 2 (Moderate)',            desc: 'Hemorrhages/exudates. Early clinical intervention.' },
              { icon: HiXCircle,           color: '#ef4444', label: 'Grade 3 & 4 (Severe/PDR)',     desc: 'Neovascularization. Immediate surgical referral required.' },
            ].map(({ icon: Icon, color, label, desc }) => (
              <li key={label} className="flex items-start gap-3 text-sm">
                <Icon className="shrink-0 mt-0.5" style={{ color }} />
                <span style={{ color: 'var(--text-secondary)' }}>
                  <strong style={{ color: 'var(--text-primary)' }}>{label}: </strong>{desc}
                </span>
              </li>
            ))}
          </ul>
        </motion.div>
      </section>
    </div>
  )
}
