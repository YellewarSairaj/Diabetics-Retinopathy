import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import MainLayout from '@/layouts/MainLayout'
import DashboardLayout from '@/layouts/DashboardLayout'
import Loader from '@/components/Loader/Loader'

// Lazy-loaded pages for code splitting & performance
const Home       = lazy(() => import('@/pages/Home'))
const Dashboard  = lazy(() => import('@/pages/Dashboard'))
const Prediction = lazy(() => import('@/pages/Prediction'))
const GradCAM    = lazy(() => import('@/pages/GradCAM'))
const Reports    = lazy(() => import('@/pages/Reports'))
const About      = lazy(() => import('@/pages/About'))
const Contact    = lazy(() => import('@/pages/Contact'))
const Settings   = lazy(() => import('@/pages/Settings'))

function PageFallback() {
  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <Loader text="Loading page..." size="md" />
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          {/* Main layout: Navbar + Footer, no sidebar */}
          <Route element={<MainLayout />}>
            <Route path="/"           element={<Home />} />
            <Route path="/prediction" element={<Prediction />} />
            <Route path="/gradcam"    element={<GradCAM />} />
            <Route path="/reports"    element={<Reports />} />
            <Route path="/about"      element={<About />} />
            <Route path="/contact"    element={<Contact />} />
            <Route path="/settings"   element={<Settings />} />
          </Route>

          {/* Dashboard layout: Navbar + Sidebar */}
          <Route element={<DashboardLayout />}>
            <Route path="/dashboard" element={<Dashboard />} />
          </Route>

          {/* Fallback: redirect unknown routes to home */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  )
}
