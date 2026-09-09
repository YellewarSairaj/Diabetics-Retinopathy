import { motion } from 'framer-motion'
import {
  HiCheckCircle, HiDatabase, HiCode, HiChip, HiLightningBolt,
  HiBeaker, HiShieldCheck, HiDocumentText,
} from 'react-icons/hi'
import { RiEyeLine } from 'react-icons/ri'

import { METRICS } from '@/constants/metrics'

const FADE = (delay = 0) => ({
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, delay },
})

const TECH_STACK = [
  { label: 'Model',        value: 'EfficientNet-B0 (TensorFlow / Keras)',      icon: HiChip,          color: '#3b82f6' },
  { label: 'Dataset',      value: 'APTOS 2019 Blindness Detection (3,662)',    icon: HiDatabase,      color: '#22c55e' },
  { label: 'XAI',          value: 'Grad-CAM & Grad-CAM++ (top_activation)',   icon: HiLightningBolt, color: '#f59e0b' },
  { label: 'Backend',      value: 'FastAPI + SQLAlchemy + SQLite',              icon: HiCode,          color: '#8b5cf6' },
  { label: 'Frontend',     value: 'React 19 + Vite + Tailwind CSS',            icon: HiBeaker,        color: '#06b6d4' },
  { label: 'Augmentation', value: 'Albumentations + Ben Graham Preprocessing', icon: HiShieldCheck,   color: '#f97316' },
]

export default function About() {
  return (
    <div className="flex flex-col gap-10">
      {/* Header */}
      <motion.div {...FADE(0)}>
        <h1 className="text-2xl font-bold" style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
          About RetinaX AI Platform
        </h1>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          Clinical deep learning research architecture &amp; technical specification
        </p>
      </motion.div>

      {/* Hero description */}
      <motion.div {...FADE(0.05)} className="glass-card">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: 'linear-gradient(135deg,#2563eb,#7c3aed)' }}>
            <RiEyeLine className="text-white text-xl" />
          </div>
          <div>
            <h2 className="font-bold text-lg" style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
              RetinaX AI — Enterprise Explainable Ophthalmic Intelligence
            </h2>
            <p className="text-sm mt-2 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
              RetinaX AI is a production-grade ophthalmic AI decision support system designed for clinical diabetic
              retinopathy screening. It combines state-of-the-art transfer learning with Gradient-weighted Class
              Activation Mapping (Grad-CAM) to provide both accurate DR severity classification and visually
              interpretable explanations for each prediction — enabling clinicians to trust and validate every AI-driven decision.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Two-column cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Architecture */}
        <motion.div {...FADE(0.1)} className="glass-card">
          <h3 className="font-bold text-base flex items-center gap-2 mb-4"
            style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
            <HiChip style={{ color: '#3b82f6' }} /> Model Architecture
          </h3>
          <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
            Utilizes EfficientNet-B0 transfer learning with ImageNet pre-trained weights and two-stage fine-tuning:
            Stage 1 trains only the custom classification head, Stage 2 unfreezes the last 30 layers.
          </p>
          <ul className="flex flex-col gap-3">
            {[
              'Input Resolution: 224 × 224 × 3 (RGB)',
              'Feature Head: Global Average Pooling + Dropout (0.4)',
              'Output: 5-class Softmax (No DR → Proliferative DR)',
              'XAI Target Layer: top_activation',
              'Total Parameters: ~4.06M',
            ].map(item => (
              <li key={item} className="flex items-start gap-2 text-sm">
                <HiCheckCircle className="text-green-500 mt-0.5 shrink-0" />
                <span style={{ color: 'var(--text-secondary)' }}>{item}</span>
              </li>
            ))}
          </ul>
        </motion.div>

        {/* Dataset & Training */}
        <motion.div {...FADE(0.15)} className="glass-card">
          <h3 className="font-bold text-base flex items-center gap-2 mb-4"
            style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
            <HiDatabase style={{ color: '#22c55e' }} /> Dataset &amp; Training
          </h3>
          <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
            APTOS 2019 Blindness Detection dataset containing 3,662 validated fundus photographs across 5 severity grades,
            with significant class imbalance (9.35:1 between Grade 0 and Grade 4).
          </p>
          <ul className="flex flex-col gap-3">
            {[
              '70% Train (2,562) · 15% Val (550) · 15% Test (550)',
              'Balanced Class Weights for 9.35:1 imbalance mitigation',
              'Ben Graham Local Color Subtraction (σₓ = 30)',
              'Augmentation: RandomCrop, Flip, Brightness, Contrast',
              'Optimizer: Adam (LR 1e-4 → 1e-5 with ReduceLROnPlateau)',
            ].map(item => (
              <li key={item} className="flex items-start gap-2 text-sm">
                <HiCheckCircle className="text-green-500 mt-0.5 shrink-0" />
                <span style={{ color: 'var(--text-secondary)' }}>{item}</span>
              </li>
            ))}
          </ul>
        </motion.div>
      </div>

      {/* Technology Stack */}
      <motion.div {...FADE(0.2)} className="glass-card">
        <h3 className="font-bold text-base flex items-center gap-2 mb-6"
          style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
          <HiCode style={{ color: 'var(--primary)' }} /> Full Technology Stack
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {TECH_STACK.map(({ label, value, icon: Icon, color }) => (
            <div key={label} className="flex items-start gap-3 rounded-xl p-4"
              style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-color)' }}>
              <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                style={{ background: `${color}18`, border: `1px solid ${color}30` }}>
                <Icon style={{ color }} />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>{label}</p>
                <p className="text-sm font-medium mt-0.5" style={{ color: 'var(--text-primary)' }}>{value}</p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Performance Results */}
      <motion.div {...FADE(0.25)} className="glass-card">
        <h3 className="font-bold text-base flex items-center gap-2 mb-6"
          style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
          <HiDocumentText style={{ color: '#f59e0b' }} /> Evaluation Results (Test Set)
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: 'Accuracy',    value: METRICS.accuracy, color: '#22c55e' },
            { label: 'Precision',   value: METRICS.precision, color: '#3b82f6' },
            { label: 'Recall',      value: METRICS.recall, color: '#8b5cf6' },
            { label: 'ROC-AUC',     value: METRICS.rocAuc, color: '#f59e0b' },
          ].map(({ label, value, color }) => (
            <div key={label} className="rounded-xl p-4 text-center"
              style={{ background: `${color}0f`, border: `1px solid ${color}25` }}>
              <p className="text-2xl font-bold" style={{ color, fontFamily: 'Poppins,sans-serif' }}>{value}</p>
              <p className="text-xs mt-1 font-medium" style={{ color: 'var(--text-secondary)' }}>{label}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}
