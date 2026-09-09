import { NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  HiHome, HiChartBar, HiUpload, HiEye, HiDocumentText,
  HiInformationCircle, HiMail, HiCog,
} from 'react-icons/hi'
import { RiEyeLine } from 'react-icons/ri'

const SIDEBAR_ITEMS = [
  { path: '/',           label: 'Home',        icon: HiHome,              group: 'main' },
  { path: '/dashboard',  label: 'Dashboard',   icon: HiChartBar,          group: 'main' },
  { path: '/prediction', label: 'Prediction',  icon: HiUpload,            group: 'tools' },
  { path: '/gradcam',    label: 'Grad-CAM XAI',icon: HiEye,              group: 'tools' },
  { path: '/reports',    label: 'Reports',     icon: HiDocumentText,      group: 'tools' },
  { path: '/about',      label: 'About',       icon: HiInformationCircle, group: 'info' },
  { path: '/contact',    label: 'Contact',     icon: HiMail,              group: 'info' },
  { path: '/settings',   label: 'Settings',    icon: HiCog,               group: 'info' },
]

const GROUP_LABELS = { main: 'Overview', tools: 'Clinical Tools', info: 'Information' }

export default function Sidebar() {
  const grouped = SIDEBAR_ITEMS.reduce((acc, item) => {
    if (!acc[item.group]) acc[item.group] = []
    acc[item.group].push(item)
    return acc
  }, {})

  return (
    <aside
      className="fixed left-0 top-16 bottom-0 z-30 w-56 flex flex-col py-6 px-3 overflow-y-auto"
      style={{
        background: 'var(--bg-secondary)',
        borderRight: '1px solid var(--border-color)',
        backdropFilter: 'blur(20px)',
      }}
    >
      {Object.entries(grouped).map(([group, items]) => (
        <div key={group} className="mb-6">
          <p className="px-3 mb-2 text-xs font-semibold uppercase tracking-widest"
            style={{ color: 'var(--text-muted)' }}>
            {GROUP_LABELS[group]}
          </p>
          <ul className="flex flex-col gap-0.5">
            {items.map(({ path, label, icon: Icon }) => (
              <li key={path}>
                <NavLink
                  to={path}
                  end={path === '/'}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group"
                  style={({ isActive }) => ({
                    background: isActive ? 'linear-gradient(135deg,rgba(37,99,235,0.2),rgba(79,70,229,0.15))' : 'transparent',
                    color: isActive ? 'var(--primary)' : 'var(--text-secondary)',
                    borderLeft: isActive ? '3px solid var(--primary)' : '3px solid transparent',
                  })}
                >
                  <Icon className="text-base shrink-0" />
                  <span>{label}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}

      {/* Footer branding */}
      <div className="mt-auto px-3">
        <div className="flex items-center gap-2 p-3 rounded-xl"
          style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-color)' }}>
          <div className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg,#2563eb,#7c3aed)' }}>
            <RiEyeLine className="text-white text-sm" />
          </div>
          <div>
            <p className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>RetinaX AI</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>v1.0.0 Enterprise</p>
          </div>
        </div>
      </div>
    </aside>
  )
}
