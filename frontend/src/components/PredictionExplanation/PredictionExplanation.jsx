import { motion } from 'framer-motion'
import { HiSparkles, HiStar, HiInformationCircle, HiShieldCheck } from 'react-icons/hi'
import { SEVERITY_COLORS, getConfidenceLevel } from '@/utils/constants'

/**
 * Dynamically computes non-prescriptive, score-based explanation text.
 * Never fabricates clinical lesion findings (e.g. hemorrhages/exudates).
 */
function getWhyThisPredictionText(predictedClass = 'No DR', confidenceScore = 0) {
  const confPct = (confidenceScore * 100).toFixed(1)
  const confMeta = getConfidenceLevel(confidenceScore)

  if (confMeta.level === 'high') {
    return `The model assigned the highest score to ${predictedClass} (${confPct}%), which is substantially higher than the scores assigned to the other severity classes. Therefore, ${predictedClass} is selected as the model's predicted class.`
  }

  if (confMeta.level === 'moderate') {
    return `The model assigned the highest score to ${predictedClass} (${confPct}%), which is higher than the scores assigned to the other severity classes. Therefore, ${predictedClass} is selected as the model's predicted class.`
  }

  return `The model assigned the highest score to ${predictedClass} (${confPct}%), but the scores of the other classes are relatively close. This indicates that the model is uncertain about this prediction.`
}

/**
 * Dynamically computes safe clinical decision support text based on Top Model Score.
 */
function getClinicalDecisionSupportText(predictedClass = 'No DR', confidenceScore = 0) {
  const confPct = (confidenceScore * 100).toFixed(1)
  const confMeta = getConfidenceLevel(confidenceScore)

  if (confMeta.level === 'high') {
    return `The AI result indicates ${predictedClass} with a high model score (${confPct}%). However, this result should be interpreted by a qualified ophthalmologist and should not be considered a standalone medical diagnosis.`
  }

  if (confMeta.level === 'moderate') {
    return `The AI result indicates ${predictedClass} with a moderate model score (${confPct}%). This screening output should be evaluated alongside clinical examination by a qualified ophthalmologist.`
  }

  return `The AI result indicates ${predictedClass}, but the model score is low (${confPct}%). The result should be reviewed and confirmed by a qualified ophthalmologist. This AI system does not independently determine diagnosis, treatment, or surgical requirements.`
}

function ProbabilityRow({ item, isTarget }) {
  const color = SEVERITY_COLORS[item.class_index] ?? '#3b82f6'
  const pct = (item.probability * 100).toFixed(1)

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-sm">
        <span
          className="flex items-center gap-1.5 font-medium"
          style={{ color: isTarget ? color : 'var(--text-secondary)' }}
        >
          {isTarget && <HiStar className="text-xs" />}
          {item.class_name}
        </span>
        <span
          className="font-semibold tabular-nums"
          style={{ color: isTarget ? color : 'var(--text-secondary)' }}
        >
          {pct}%
        </span>
      </div>
      <div className="prob-track">
        <motion.div
          className="prob-fill"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, delay: item.class_index * 0.08, ease: [0.34, 1.56, 0.64, 1] }}
          style={{ background: color, opacity: isTarget ? 1 : 0.45 }}
        />
      </div>
    </div>
  )
}

/**
 * PredictionExplanation Component
 * Professional card displaying AI Prediction Summary, Top Model Score,
 * Probability Distribution, Dynamic "Why this prediction?", and Decision Support.
 */
export default function PredictionExplanation({
  predictedClass = 'No DR',
  classProbabilities = [],
  confidence = 0,
  confidenceStatus,
}) {
  const confScore = Number(confidence) || 0
  const confPct = (confScore * 100).toFixed(1)
  const confMeta = getConfidenceLevel(confScore)
  const statusLabel = confidenceStatus || confMeta.status
  const predictedIdx = classProbabilities.findIndex(
    (p) => p.class_name?.toLowerCase() === predictedClass?.toLowerCase()
  )
  const targetColor = SEVERITY_COLORS[predictedIdx >= 0 ? predictedIdx : 0] ?? '#3b82f6'

  const whyText = getWhyThisPredictionText(predictedClass, confScore)
  const clinicalSupportText = getClinicalDecisionSupportText(predictedClass, confScore)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="glass-card flex flex-col gap-6"
    >
      {/* Header */}
      <div
        className="flex items-center justify-between pb-4"
        style={{ borderBottom: '1px solid var(--border-color)' }}
      >
        <h3
          className="font-bold text-base flex items-center gap-2"
          style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}
        >
          <HiSparkles className="text-blue-500 text-lg" />
          AI Prediction Summary
        </h3>
        <span
          className="text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider"
          style={{
            background: confMeta.bgColor,
            color: confMeta.color,
            border: `1px solid ${confMeta.borderColor}`,
          }}
        >
          {statusLabel}
        </span>
      </div>

      {/* Main Score Metrics Banner */}
      <div
        className="rounded-xl p-5"
        style={{
          background: `linear-gradient(135deg, ${targetColor}15, ${targetColor}08)`,
          border: `1px solid ${targetColor}40`,
          borderLeft: `4px solid ${targetColor}`,
        }}
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 items-center">
          <div>
            <span
              className="text-xs uppercase tracking-widest font-semibold"
              style={{ color: 'var(--text-muted)' }}
            >
              1. Predicted Class
            </span>
            <h4
              className="text-2xl font-extrabold mt-0.5"
              style={{ color: targetColor, fontFamily: 'Poppins,sans-serif' }}
            >
              {predictedClass}
            </h4>
          </div>

          <div className="flex flex-col sm:items-end">
            <span
              className="text-xs uppercase tracking-widest font-semibold"
              style={{ color: 'var(--text-muted)' }}
            >
              2. Top Model Score
            </span>
            <div
              className="text-3xl font-extrabold tabular-nums mt-0.5"
              style={{ color: targetColor, fontFamily: 'Poppins,sans-serif' }}
            >
              {confPct}%
            </div>
          </div>
        </div>
      </div>

      {/* Probability Distribution Section */}
      <div>
        <h4
          className="text-xs font-semibold uppercase tracking-wider mb-3"
          style={{ color: 'var(--text-muted)' }}
        >
          4. Probability Distribution
        </h4>
        <div className="flex flex-col gap-3">
          {classProbabilities.map((item, idx) => (
            <ProbabilityRow
              key={item.class_name || idx}
              item={item}
              isTarget={
                item.class_name?.toLowerCase() === predictedClass?.toLowerCase() ||
                item.class_index === predictedIdx
              }
            />
          ))}
        </div>
      </div>

      {/* Why this prediction? Section */}
      <div
        className="rounded-xl p-4 flex flex-col gap-2"
        style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-color)' }}
      >
        <h4
          className="text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5"
          style={{ color: 'var(--text-primary)' }}
        >
          <HiInformationCircle className="text-blue-500 text-sm" />
          5. Why this prediction?
        </h4>
        <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          {whyText}
        </p>
      </div>

      {/* Clinical Decision Support Section */}
      <div
        className="rounded-xl p-4 flex flex-col gap-2"
        style={{
          background: 'rgba(37, 99, 235, 0.08)',
          border: '1px solid rgba(37, 99, 235, 0.25)',
        }}
      >
        <h4
          className="text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5"
          style={{ color: 'var(--primary)' }}
        >
          <HiShieldCheck className="text-sm" />
          6. Clinical Decision Support
        </h4>
        <p className="text-sm leading-relaxed font-medium" style={{ color: 'var(--text-primary)' }}>
          {clinicalSupportText}
        </p>
      </div>
    </motion.div>
  )
}
