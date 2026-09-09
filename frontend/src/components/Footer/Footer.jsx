import { RiEyeLine } from 'react-icons/ri'

export default function Footer() {
  const year = new Date().getFullYear()
  return (
    <footer
      className="relative z-10 mt-auto py-6 px-8"
      style={{ borderTop: '1px solid var(--border-color)' }}
    >
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-sm"
        style={{ color: 'var(--text-muted)' }}>
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-md flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg,#2563eb,#7c3aed)' }}>
            <RiEyeLine className="text-white text-xs" />
          </div>
          <span>
            © {year}{' '}
            <span className="font-semibold" style={{ color: 'var(--text-secondary)' }}>RetinaX AI Platform</span>
            {' '}— Enterprise Explainable Ophthalmic Healthcare
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span>Built for Research &amp; Clinical Decision Support</span>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-colors hover:underline"
            style={{ color: 'var(--primary)' }}
          >
            API Docs
          </a>
        </div>
      </div>
    </footer>
  )
}
