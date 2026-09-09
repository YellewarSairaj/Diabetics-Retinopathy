import { useCallback, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { HiUpload, HiX, HiPhotograph } from 'react-icons/hi'
import { usePrediction } from '@/hooks/usePrediction'

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff']
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const SAMPLES = [
  { url: `${API_BASE}/outputs/preprocessing_comparison.png`, title: 'Sample 1 — Preprocessed' },
  { url: `${API_BASE}/outputs/gradcam_sample_overlay.png`,   title: 'Sample 2 — Grad-CAM' },
]

export default function UploadBox() {
  const { selectFile, previewUrl, clearPrediction, selectedFile } = usePrediction()
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef(null)

  const MAX_FILE_SIZE = 15 * 1024 * 1024 // 15MB limit

  const handleFile = useCallback((file) => {
    if (!file) return
    const isImage = (file.type && file.type.startsWith('image/')) || /\.(png|jpe?g|bmp|tiff|webp)$/i.test(file.name)
    if (!isImage) {
      setError('Invalid file: Please upload a valid retinal fundus image (.png, .jpg).')
      return
    }
    if (file.size > MAX_FILE_SIZE) {
      setError('Invalid file: Image size exceeds 15MB limit. Please upload a smaller file.')
      return
    }
    setError('')
    selectFile(file)
  }, [selectFile])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }, [handleFile])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const onInputChange = (e) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const loadSample = async (url) => {
    try {
      const res = await fetch(url)
      const blob = await res.blob()
      const name = url.split('/').pop() || 'sample_fundus.png'
      const file = new File([blob], name, { type: 'image/png' })
      handleFile(file)
    } catch {
      setError('Failed to load sample image. Ensure backend is running.')
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h3 className="text-base font-semibold flex items-center gap-2 mb-1"
          style={{ color: 'var(--text-primary)', fontFamily: 'Poppins,sans-serif' }}>
          <HiUpload className="text-blue-500" />
          Upload Fundus Photograph
        </h3>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          Select or drag a retinal fundus image (.png, .jpg, .jpeg)
        </p>
      </div>

      {/* Dropzone */}
      <motion.div
        onClick={() => !previewUrl && fileInputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        animate={{ scale: dragging ? 1.01 : 1 }}
        className={`relative rounded-2xl overflow-hidden transition-all duration-300 ${!previewUrl ? 'cursor-pointer' : ''} ${dragging ? 'dropzone-active' : ''}`}
        style={{
          minHeight: '220px',
          border: `2px dashed ${dragging ? 'var(--primary)' : 'var(--border-color)'}`,
          background: dragging ? 'rgba(37,99,235,0.06)' : 'var(--bg-glass)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,.png,.jpg,.jpeg,.bmp,.tiff"
          className="hidden"
          onChange={onInputChange}
          id="fundus-file-input"
        />

        <AnimatePresence mode="wait">
          {!previewUrl ? (
            <motion.div
              key="prompt"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex flex-col items-center gap-3 p-8 text-center"
            >
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center"
                style={{ background: 'rgba(37,99,235,0.12)', border: '1px solid rgba(37,99,235,0.25)' }}>
                <HiPhotograph className="text-3xl" style={{ color: 'var(--primary)' }} />
              </div>
              <div>
                <p className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
                  Drag &amp; Drop Retinal Image Here
                </p>
                <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                  or click to browse files
                </p>
              </div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                PNG · JPG · JPEG · BMP · TIFF supported
              </p>
            </motion.div>
          ) : (
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="relative w-full"
            >
              <img
                src={previewUrl}
                alt="Selected fundus photograph"
                className="w-full object-contain rounded-2xl"
                style={{ maxHeight: '280px' }}
              />
              {/* Remove button */}
              <button
                onClick={(e) => { e.stopPropagation(); clearPrediction() }}
                className="absolute top-3 right-3 w-8 h-8 rounded-full flex items-center justify-center transition-all hover:scale-110"
                style={{ background: 'rgba(239,68,68,0.9)', color: '#fff' }}
                title="Remove image"
                aria-label="Remove image"
              >
                <HiX className="text-base" />
              </button>
              {/* File name badge */}
              <div className="absolute bottom-3 left-3 px-3 py-1 rounded-full text-xs font-medium"
                style={{ background: 'rgba(0,0,0,0.7)', color: '#fff', backdropFilter: 'blur(6px)' }}>
                {selectedFile?.name}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="text-sm px-3 py-2 rounded-lg"
            style={{ background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.25)' }}
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>

      {/* Sample images */}
      <div>
        <p className="text-xs mb-2" style={{ color: 'var(--text-secondary)' }}>
          Or select a pre-loaded sample:
        </p>
        <div className="flex gap-3">
          {SAMPLES.map((s, i) => (
            <button
              key={i}
              onClick={() => loadSample(s.url)}
              title={s.title}
              className="relative w-16 h-16 rounded-xl overflow-hidden border-2 transition-all hover:scale-105"
              style={{ borderColor: 'var(--border-color)', background: 'var(--bg-glass)' }}
            >
              <img
                src={s.url}
                alt={s.title}
                className="w-full h-full object-cover"
                onError={(e) => { e.target.style.display = 'none' }}
              />
              <div className="absolute inset-0 flex items-center justify-center"
                style={{ background: 'rgba(0,0,0,0.4)', opacity: 0 }}
                onMouseEnter={e => e.currentTarget.style.opacity = 1}
                onMouseLeave={e => e.currentTarget.style.opacity = 0}
              >
                <span className="text-white text-xs">#{i + 1}</span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
