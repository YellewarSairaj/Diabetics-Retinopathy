/** DR severity color palette */
export const SEVERITY_COLORS = [
  '#22c55e', // Grade 0: No DR
  '#3b82f6', // Grade 1: Mild
  '#eab308', // Grade 2: Moderate
  '#f97316', // Grade 3: Severe
  '#ef4444', // Grade 4: Proliferative DR
]

export const SEVERITY_LABELS = [
  'No DR',
  'Mild',
  'Moderate',
  'Severe',
  'Proliferative DR',
]

/** Non-prescriptive decision support guidelines (no lesion inferencing or treatment prescribing) */
export const SEVERITY_DESCRIPTIONS = {
  'No DR': 'No clear signs of retinal microvascular abnormalities observed on current fundus image.',
  'Mild': 'Classified as Mild DR grade. Routine ophthalmic evaluation recommended.',
  'Moderate': 'Classified as Moderate DR grade. Clinical evaluation recommended.',
  'Severe': 'Classified as Severe DR grade. Specialist ophthalmic evaluation recommended.',
  'Proliferative DR': 'Classified as Proliferative DR grade. Specialist ophthalmic evaluation recommended to determine clinical status.',
}

/** Global Medical Safety Disclaimers */
export const MEDICAL_DISCLAIMER =
  'This AI system is intended for research and clinical decision support. It is not a substitute for professional medical diagnosis.'

export const GRADCAM_CAPTION =
  "Highlighted regions indicate image areas that contributed to the model's prediction."

export const GRADCAM_DISCLAIMER =
  'Grad-CAM is an AI interpretability visualization and should not be considered a clinical lesion segmentation.'

/**
 * Evaluates model confidence float (0.0 to 1.0) and returns threshold metadata.
 */
export function getConfidenceLevel(confidence = 0) {
  const score = Number(confidence) || 0

  if (score >= 0.8) {
    return {
      status: 'HIGH CONFIDENCE',
      level: 'high',
      color: '#22c55e',
      bgColor: 'rgba(34, 197, 94, 0.15)',
      borderColor: 'rgba(34, 197, 94, 0.35)',
      recommendationText: 'High Confidence Prediction. For clinical decision support only.',
    }
  }

  if (score >= 0.5) {
    return {
      status: 'MODERATE CONFIDENCE',
      level: 'moderate',
      color: '#eab308',
      bgColor: 'rgba(234, 179, 8, 0.15)',
      borderColor: 'rgba(234, 179, 8, 0.35)',
      recommendationText: 'Moderate Confidence Prediction. Review alongside clinical evaluation.',
    }
  }

  return {
    status: 'LOW CONFIDENCE',
    level: 'low',
    color: '#ef4444',
    bgColor: 'rgba(239, 68, 68, 0.15)',
    borderColor: 'rgba(239, 68, 68, 0.35)',
    recommendationText:
      'Low Confidence Prediction. AI prediction is uncertain. Clinical review by a qualified ophthalmologist is strictly recommended.',
  }
}

/**
 * Dynamic Clinical Decision Support Helper
 * Computes non-prescriptive, safe decision support text strictly based on model output values.
 * Never infers specific retinal lesions or prescribes treatment/surgery.
 */
export function getDynamicClinicalSupport(predictedClass = 'No DR', confidence = 0) {
  const confScore = Number(confidence) || 0
  const confPct = (confScore * 100).toFixed(1)
  const confMeta = getConfidenceLevel(confScore)

  if (confMeta.level === 'low') {
    return `The model's highest-probability classification is ${predictedClass}, but the prediction confidence is low (${confPct}%). The result should be reviewed and confirmed by a qualified ophthalmologist. This AI system does not independently determine diagnosis, treatment, or surgical requirements.`
  }

  if (confMeta.level === 'moderate') {
    return `The model's highest-probability classification is ${predictedClass} with moderate confidence (${confPct}%). This screening output should be evaluated alongside clinical examination by a qualified eye-care professional. This AI system does not independently determine treatment or surgical requirements.`
  }

  return `The model's highest-probability classification is ${predictedClass} with high confidence (${confPct}%). This screening output is provided for clinical decision support only. Final diagnostic evaluation and management decisions remain the responsibility of a qualified ophthalmologist.`
}

/** Nav route definitions */
export const NAV_ROUTES = [
  { path: '/',           label: 'Home',       icon: 'HiHome' },
  { path: '/dashboard',  label: 'Dashboard',  icon: 'HiChartBar' },
  { path: '/prediction', label: 'Prediction', icon: 'HiScan' },
  { path: '/gradcam',    label: 'Grad-CAM',   icon: 'HiEye' },
  { path: '/reports',    label: 'Reports',    icon: 'HiDocumentText' },
  { path: '/about',      label: 'About',      icon: 'HiInformationCircle' },
  { path: '/contact',    label: 'Contact',    icon: 'HiMail' },
]

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
