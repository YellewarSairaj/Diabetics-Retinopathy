import { motion } from 'framer-motion'

export default function Loader({ text = 'Processing...', size = 'md', fullScreen = false }) {
  const sizes = { sm: 'w-6 h-6 border-2', md: 'w-10 h-10 border-3', lg: 'w-16 h-16 border-4' }

  const spinner = (
    <div className="flex flex-col items-center gap-4">
      <div className="relative">
        {/* Outer ring */}
        <motion.div
          className={`${sizes[size]} rounded-full`}
          style={{ border: '3px solid var(--border-color)' }}
          animate={{ rotate: 360 }}
          transition={{ duration: 1.2, repeat: Infinity, ease: 'linear' }}
        />
        {/* Inner glow arc */}
        <motion.div
          className={`absolute inset-0 ${sizes[size]} rounded-full`}
          style={{
            border: '3px solid transparent',
            borderTopColor: 'var(--primary)',
            borderRightColor: 'var(--primary)',
            filter: 'drop-shadow(0 0 6px var(--primary))',
          }}
          animate={{ rotate: 360 }}
          transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}
        />
      </div>
      {text && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="text-sm font-medium"
          style={{ color: 'var(--text-secondary)' }}
        >
          {text}
        </motion.p>
      )}
    </div>
  )

  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center"
        style={{ background: 'rgba(6,11,24,0.8)', backdropFilter: 'blur(8px)' }}>
        {spinner}
      </div>
    )
  }

  return (
    <div className="flex items-center justify-center py-12">
      {spinner}
    </div>
  )
}
