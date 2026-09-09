import { motion } from 'framer-motion'
import { HiSun, HiMoon } from 'react-icons/hi'
import { useTheme } from '@/hooks/useTheme'

export default function ThemeSwitcher({ className = '' }) {
  const { theme, toggleTheme } = useTheme()
  const isDark = theme === 'dark'

  return (
    <motion.button
      onClick={toggleTheme}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
      aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
      className={`relative w-9 h-9 rounded-xl flex items-center justify-center transition-colors ${className}`}
      style={{
        background: 'var(--bg-glass)',
        border: '1px solid var(--border-color)',
        color: isDark ? '#fbbf24' : '#475569',
      }}
    >
      <motion.div
        key={theme}
        initial={{ rotate: -90, opacity: 0, scale: 0.5 }}
        animate={{ rotate: 0, opacity: 1, scale: 1 }}
        exit={{ rotate: 90, opacity: 0, scale: 0.5 }}
        transition={{ duration: 0.2 }}
      >
        {isDark ? <HiSun className="text-lg" /> : <HiMoon className="text-lg" />}
      </motion.div>
    </motion.button>
  )
}
