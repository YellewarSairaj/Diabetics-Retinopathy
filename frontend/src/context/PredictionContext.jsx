import { createContext, useContext, useState, useCallback, useRef } from 'react'
import { apiService } from '@/services/api'

const PredictionContext = createContext(null)

/**
 * Task 4: Strict Probability Validation Logic
 * Validates API response data payload before committing to React state.
 */
function validatePredictionPayload(data) {
  if (!data || !data.prediction) {
    throw new Error('Analysis failed: Received empty response from diagnostic backend.')
  }

  const { prediction } = data
  const { class_probabilities, confidence, predicted_index, predicted_class } = prediction

  if (!Array.isArray(class_probabilities) || class_probabilities.length !== 5) {
    throw new Error('Analysis failed: Invalid class probability distribution returned.')
  }

  let sumProbs = 0
  let maxProb = -1
  let maxIdx = -1

  class_probabilities.forEach((item, idx) => {
    const prob = Number(item.probability)
    if (isNaN(prob) || prob < 0 || prob > 1) {
      throw new Error(`Analysis failed: Invalid probability value (${item.probability}) for class '${item.class_name}'.`)
    }
    sumProbs += prob
    if (prob > maxProb) {
      maxProb = prob
      maxIdx = idx
    }
  })

  // Check sum of probabilities (allow standard float tolerance)
  if (Math.abs(sumProbs - 1.0) > 0.05) {
    throw new Error(`Analysis failed: Class probabilities sum to ${sumProbs.toFixed(4)}, expected 1.0.`)
  }

  // Check confidence matches highest probability
  if (Math.abs(confidence - maxProb) > 0.01) {
    throw new Error(`Analysis failed: Confidence (${confidence}) does not match highest probability (${maxProb}).`)
  }

  // Check predicted_index matches index of max probability
  if (predicted_index !== maxIdx) {
    throw new Error(`Analysis failed: Predicted index (${predicted_index}) does not correspond to highest probability class index (${maxIdx}).`)
  }

  return true
}

export function PredictionProvider({ children }) {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [explainData, setExplainData] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)

  const selectFile = useCallback((file) => {
    if (!file) return
    setSelectedFile(file)
    setExplainData(null)
    setError(null)
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }, [])

  const clearPrediction = useCallback(() => {
    setSelectedFile(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    setExplainData(null)
    setError(null)
  }, [previewUrl])

  const runAnalysis = useCallback(async (algorithm = 'gradcam') => {
    if (!selectedFile) return
    setIsLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const data = await apiService.postExplain(formData, algorithm)
      
      // Task 4: Validate probability distribution before rendering
      validatePredictionPayload(data)

      setExplainData(data)
      return data
    } catch (err) {
      let msg = 'Analysis failed: The model was unable to process this image. Please try another sample.'
      if (err?.code === 'ECONNABORTED' || err?.code === 'ERR_NETWORK' || err?.message?.includes('Network Error')) {
        msg = 'Connection error: The AI backend service is currently unreachable.'
      } else if (err?.response?.data?.detail) {
        msg = `Analysis failed: ${err.response.data.detail}`
      } else if (err?.message) {
        msg = err.message
      }

      setError(msg)
      setExplainData(null)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [selectedFile])

  return (
    <PredictionContext.Provider value={{
      selectedFile,
      previewUrl,
      explainData,
      isLoading,
      error,
      selectFile,
      clearPrediction,
      runAnalysis,
    }}>
      {children}
    </PredictionContext.Provider>
  )
}

export function usePrediction() {
  const ctx = useContext(PredictionContext)
  if (!ctx) throw new Error('usePrediction must be used inside PredictionProvider')
  return ctx
}

export default PredictionContext
