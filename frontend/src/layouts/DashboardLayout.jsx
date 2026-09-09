import { Outlet } from 'react-router-dom'
import Navbar from '@/components/Navbar/Navbar'
import Sidebar from '@/components/Sidebar/Sidebar'
import Footer from '@/components/Footer/Footer'

export default function DashboardLayout() {
  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      <div className="glow-orb orb-1" />
      <div className="glow-orb orb-2" />

      <Navbar />
      <Sidebar />

      {/* Content shifted right to accommodate sidebar */}
      <main className="flex-1 pt-16 pl-0 md:pl-56 page-wrapper">
        <div className="px-4 sm:px-6 lg:px-8 py-8">
          <Outlet />
        </div>
      </main>

      {/* Footer also shifted */}
      <div className="pl-0 md:pl-56">
        <Footer />
      </div>
    </div>
  )
}
