import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ThemeProvider } from '@/context/ThemeContext'
import { PredictionProvider } from '@/context/PredictionContext'
import App from './App.jsx'
import '@/assets/styles/index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      <PredictionProvider>
        <App />
      </PredictionProvider>
    </ThemeProvider>
  </StrictMode>
)
