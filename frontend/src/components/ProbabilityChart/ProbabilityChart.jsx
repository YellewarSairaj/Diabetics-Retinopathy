import { useEffect, useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend,
} from 'chart.js'
import { Bar } from 'react-chartjs-2'
import { SEVERITY_COLORS, SEVERITY_LABELS } from '@/utils/constants'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

export default function ProbabilityChart({ probabilities = [], predictedIndex = -1 }) {
  if (!probabilities.length) return null

  const values = probabilities.map(p => parseFloat((p.probability * 100).toFixed(2)))
  const backgroundColors = SEVERITY_COLORS.map((c, i) =>
    i === predictedIndex ? c : `${c}55`
  )
  const borderColors = SEVERITY_COLORS.map((c, i) =>
    i === predictedIndex ? c : `${c}88`
  )

  const data = {
    labels: SEVERITY_LABELS,
    datasets: [
      {
        label: 'Probability (%)',
        data: values,
        backgroundColor: backgroundColors,
        borderColor: borderColors,
        borderWidth: 2,
        borderRadius: 8,
        borderSkipped: false,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(13,25,50,0.95)',
        titleColor: '#f0f4ff',
        bodyColor: '#94a3b8',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        padding: 12,
        callbacks: {
          label: (ctx) => ` ${ctx.parsed.y.toFixed(2)}%`,
        },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#94a3b8', font: { size: 11 } },
      },
      y: {
        min: 0,
        max: 100,
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: {
          color: '#94a3b8',
          font: { size: 11 },
          callback: v => `${v}%`,
        },
      },
    },
    animation: { duration: 900, easing: 'easeOutQuart' },
  }

  return (
    <div style={{ height: '220px' }}>
      <Bar data={data} options={options} />
    </div>
  )
}
