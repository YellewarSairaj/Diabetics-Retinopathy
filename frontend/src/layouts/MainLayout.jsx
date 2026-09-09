import { Outlet } from 'react-router-dom'
import Navbar from '@/components/Navbar/Navbar'
import Footer from '@/components/Footer/Footer'

export default function MainLayout() {
  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      {/* Ambient glow orbs */}
      <div className="glow-orb orb-1" />
      <div className="glow-orb orb-2" />

      <Navbar />

      {/* Page content pushed below fixed navbar */}
      <main className="flex-1 pt-16 page-wrapper">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Outlet />
        </div>
      </main>

      <Footer />
    </div>
  )
}
