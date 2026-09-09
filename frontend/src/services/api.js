import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const axiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 120000, // 2 min for heavy ML inference
  headers: { 'Accept': 'application/json' },
})

// Response interceptor for error normalization
axiosInstance.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const detail = err?.response?.data?.detail || err.message || 'Request failed'
    return Promise.reject({ ...err, message: detail })
  }
)

export const apiService = {
  /** GET /health */
  getHealth: () => axiosInstance.get('/health'),

  /** GET /metrics */
  getMetrics: () => axiosInstance.get('/metrics'),

  /** GET /model-info */
  getModelInfo: () => axiosInstance.get('/model-info'),

  /** GET /api/v1/history?limit=N */
  getPredictionHistory: (limit = 20) =>
    axiosInstance.get(`/api/v1/history?limit=${limit}`),

  /** POST /api/v1/predict — multipart/form-data */
  postPredict: (formData) =>
    axiosInstance.post('/api/v1/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),

  /** POST /api/v1/explain — multipart/form-data + optional algorithm param */
  postExplain: (formData, algorithm = 'gradcam') =>
    axiosInstance.post(`/api/v1/explain?algorithm=${algorithm}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),

  /** POST /api/v1/report/pdf — returns PDF blob */
  downloadPdfReport: (formData, algorithm = 'gradcam') =>
    axiosInstance.post(`/api/v1/report/pdf?algorithm=${algorithm}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      responseType: 'blob',
    }),
}

export default apiService
