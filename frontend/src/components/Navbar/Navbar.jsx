import { useState, useEffect } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  HiHome, HiChartBar, HiEye, HiDocumentText,
  HiInformationCircle, HiMail, HiMenu, HiX,
  HiCog, HiUpload,
} from 'react-icons/hi'
import { RiEyeLine } from 'react-icons/ri'
import { useTheme } from '@/hooks/useTheme'
import ThemeSwitcher from '@/components/ThemeSwitcher/ThemeSwitcher'
import { apiService } from '@/services/api'

const NAV_ITEMS = [
  { path: '/',           label: 'Home',       icon: HiHome },
  { path: '/dashboard',  label: 'Dashboard',  icon: HiChartBar },
  { path: '/prediction', label: 'Prediction', icon: HiUpload },
  { path: '/gradcam',    label: 'Grad-CAM',   icon: HiEye },
  { path: '/reports',    label: 'Reports',    icon: HiDocumentText },
  { path: '/about',      label: 'About',      icon: HiInformationCircle },
  { path: '/contact',    label: 'Contact',    icon: HiMail },
]

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [serverStatus, setServerStatus] = useState({ text: 'Connecting...', online: null })
  const { theme } = useTheme()
  const navigate = useNavigate()

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const data = await apiService.getHealth()
        const arch = data?.model_architecture?.split(' ')[0] || 'Online'
        setServerStatus({ text: `Online · ${arch}`, online: true })
      } catch {
        setServerStatus({ text: 'Backend Offline', online: false })
      }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <>
      <nav
        className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-3"
        style={{
          background: 'var(--navbar-bg)',
          borderBottom: '1px solid var(--border-color)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
        }}
      >
        {/* Brand */}
        <motion.div
          className="flex items-center gap-3 cursor-pointer select-none"
          onClick={() => navigate('/')}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <div className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg,#2563eb,#7c3aed)', boxShadow: '0 0 16px rgba(37,99,235,0.4)' }}>
            <RiEyeLine className="text-white text-lg" />
          </div>
          <div className="hidden sm:block">
            <div className="flex items-center gap-2">
              <span className="font-bold text-base leading-tight" style={{ fontFamily: 'Poppins, sans-serif', color: 'var(--text-primary)' }}>
                RetinaX AI
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full font-semibold"
                style={{ background: 'rgba(37,99,235,0.15)', color: 'var(--primary)', border: '1px solid rgba(37,99,235,0.3)' }}>
                Healthcare
              </span>
            </div>
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Explainable Ophthalmic Intelligence</p>
          </div>
        </motion.div>

        {/* Desktop Nav Links */}
        <ul className="hidden lg:flex items-center gap-1">
          {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
            <li key={path}>
              <NavLink
                to={path}
                end={path === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? 'text-white'
                      : 'hover:opacity-100'
                  }`
                }
                style={({ isActive }) => ({
                  background: isActive ? 'linear-gradient(135deg,#2563eb,#4f46e5)' : 'transparent',
                  color: isActive ? '#fff' : 'var(--text-secondary)',
                  boxShadow: isActive ? '0 0 12px rgba(37,99,235,0.35)' : 'none',
                })}
              >
                <Icon className="text-base" />
                {label}
              </NavLink>
            </li>
          ))}
        </ul>

        {/* Right Actions */}
        <div className="flex items-center gap-3">
          {/* Status Pill */}
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium"
            style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}>
            <span className={serverStatus.online === false ? 'status-dot-red' : 'status-dot-green'} />
            {serverStatus.text}
          </div>

          <ThemeSwitcher />

          {/* Avatar */}
          <div className="flex items-center gap-2 cursor-pointer select-none hidden sm:flex">
            <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white"
              style={{ background: 'linear-gradient(135deg,#2563eb,#7c3aed)' }}>
              DR
            </div>
            <span className="hidden md:block text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Dr. Smith</span>
          </div>

          {/* Settings Icon */}
          <NavLink to="/settings" className="p-2 rounded-lg transition-colors"
            style={{ color: 'var(--text-secondary)' }}
            title="Settings">
            <HiCog className="text-lg" />
          </NavLink>

          {/* Hamburger */}
          <button
            onClick={() => setMobileOpen(v => !v)}
            className="lg:hidden p-2 rounded-lg transition-colors"
            style={{ color: 'var(--text-primary)', background: 'var(--bg-glass)' }}
            aria-label="Toggle navigation menu"
          >
            {mobileOpen ? <HiX className="text-xl" /> : <HiMenu className="text-xl" />}
          </button>
        </div>
      </nav>

      {/* Mobile Drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            key="mobile-menu"
            initial={{ opacity: 0, x: '100%' }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: '100%' }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="fixed top-0 right-0 bottom-0 z-40 w-72 flex flex-col p-6 pt-20 lg:hidden"
            style={{ background: 'var(--bg-secondary)', borderLeft: '1px solid var(--border-color)', backdropFilter: 'blur(20px)' }}
          >
            <ul className="flex flex-col gap-1">
              {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
                <li key={path}>
                  <NavLink
                    to={path}
                    end={path === '/'}
                    onClick={() => setMobileOpen(false)}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200`
                    }
                    style={({ isActive }) => ({
                      background: isActive ? 'linear-gradient(135deg,#2563eb,#4f46e5)' : 'var(--bg-glass)',
                      color: isActive ? '#fff' : 'var(--text-secondary)',
                    })}
                  >
                    <Icon className="text-base" />
                    {label}
                  </NavLink>
                </li>
              ))}
            </ul>

            <div className="mt-auto pt-6 border-t" style={{ borderColor: 'var(--border-color)' }}>
              <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                <span className={serverStatus.online === false ? 'status-dot-red' : 'status-dot-green'} />
                {serverStatus.text}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Mobile overlay backdrop */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            key="mobile-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setMobileOpen(false)}
            className="fixed inset-0 z-30 lg:hidden"
            style={{ background: 'rgba(0,0,0,0.5)' }}
          />
        )}
      </AnimatePresence>
    </>
  )
}
