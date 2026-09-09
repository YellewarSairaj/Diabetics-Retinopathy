import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { HiMail, HiPhone, HiExternalLink, HiCheckCircle } from 'react-icons/hi'
import { RiEyeLine } from 'react-icons/ri'

const FADE = (delay = 0) => ({
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, delay },
})

export default function Contact() {
  const [form, setForm] = useState({ name: '', email: '', subject: '', message: '' })
  const [submitted, setSubmitted] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => setForm(f => ({ ...f, [e.target.name]: e.target.value }))

  const handleSubmit = (e) => {
    e.preventDefault()
    setLoading(true)
    setTimeout(() => {
      setLoading(false)
      setSubmitted(true)
    }, 1200)
  }

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <motion.div {...FADE(0)}>
        <h1 className="text-2xl font-bold" style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
          Contact &amp; Clinical Support
        </h1>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          Get in touch with the RetinaX AI development &amp; ophthalmic research team
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Info card */}
        <motion.div {...FADE(0.05)} className="glass-card flex flex-col gap-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: 'linear-gradient(135deg,#2563eb,#7c3aed)' }}>
              <RiEyeLine className="text-white text-lg" />
            </div>
            <div>
              <h3 className="font-bold text-base" style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
                Clinical Operations
              </h3>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                Research collaboration &amp; enterprise deployment
              </p>
            </div>
          </div>

          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            For research collaboration, clinical validation partnerships, or enterprise deployment inquiries,
            please reach out to our team.
          </p>

          <div className="flex flex-col gap-4">
            {[
              { icon: HiMail,     label: 'Email',              value: 'support@retinax-ai.org',   href: 'mailto:support@retinax-ai.org' },
              { icon: HiPhone,    label: 'Clinical Hotline',   value: '+1 (800) RETINA-X',        href: null },
              { icon: HiExternalLink, label: 'API Docs',       value: 'OpenAPI Documentation',    href: 'http://localhost:8000/docs' },
            ].map(({ icon: Icon, label, value, href }) => (
              <div key={label} className="flex items-center gap-3 rounded-xl p-3"
                style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-color)' }}>
                <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                  style={{ background: 'rgba(37,99,235,0.12)' }}>
                  <Icon style={{ color: 'var(--primary)' }} />
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{label}</p>
                  {href ? (
                    <a href={href} target="_blank" rel="noopener noreferrer"
                      className="text-sm font-medium hover:underline" style={{ color: 'var(--primary)' }}>
                      {value}
                    </a>
                  ) : (
                    <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{value}</p>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Platform version */}
          <div className="rounded-xl p-3 flex items-center justify-between"
            style={{ background: 'rgba(37,99,235,0.08)', border: '1px solid rgba(37,99,235,0.2)' }}>
            <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Platform Version</span>
            <span className="text-xs font-bold" style={{ color: 'var(--primary)' }}>v1.0.0 Enterprise</span>
          </div>
        </motion.div>

        {/* Inquiry form */}
        <motion.div {...FADE(0.1)} className="glass-card">
          <h3 className="font-bold text-base mb-6" style={{ fontFamily: 'Poppins,sans-serif', color: 'var(--text-primary)' }}>
            Inquiry Form
          </h3>

          <AnimatePresence mode="wait">
            {submitted ? (
              <motion.div
                key="success"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex flex-col items-center gap-4 py-12 text-center"
              >
                <div className="w-16 h-16 rounded-full flex items-center justify-center"
                  style={{ background: 'rgba(34,197,94,0.15)', border: '2px solid rgba(34,197,94,0.4)' }}>
                  <HiCheckCircle className="text-3xl text-green-500" />
                </div>
                <div>
                  <h4 className="font-semibold" style={{ color: 'var(--text-primary)' }}>Message Sent Successfully!</h4>
                  <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
                    Our team will respond within 24 hours.
                  </p>
                </div>
                <button onClick={() => { setSubmitted(false); setForm({ name:'', email:'', subject:'', message:'' }) }}
                  className="text-sm font-medium" style={{ color: 'var(--primary)' }}>
                  Send another message
                </button>
              </motion.div>
            ) : (
              <motion.form key="form" onSubmit={handleSubmit} className="flex flex-col gap-4">
                {[
                  { name: 'name',    label: 'Your Name',     placeholder: 'Dr. Jane Doe',              type: 'text' },
                  { name: 'email',   label: 'Email Address', placeholder: 'jane@hospital.org',          type: 'email' },
                  { name: 'subject', label: 'Subject',       placeholder: 'Clinical Collaboration Inquiry', type: 'text' },
                ].map(({ name, label, placeholder, type }) => (
                  <div key={name} className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>{label}</label>
                    <input
                      type={type}
                      name={name}
                      value={form[name]}
                      onChange={handleChange}
                      placeholder={placeholder}
                      required
                      className="form-input"
                      id={`contact-${name}`}
                    />
                  </div>
                ))}
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>Message</label>
                  <textarea
                    name="message"
                    value={form.message}
                    onChange={handleChange}
                    rows={4}
                    placeholder="How can we assist your clinical research?"
                    required
                    className="form-input"
                    id="contact-message"
                  />
                </div>
                <motion.button
                  type="submit"
                  disabled={loading}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold mt-2"
                  style={{
                    background: 'linear-gradient(135deg,#2563eb,#4f46e5)',
                    color: '#fff',
                    boxShadow: '0 4px 16px rgba(37,99,235,0.3)',
                    opacity: loading ? 0.7 : 1,
                  }}
                >
                  {loading ? (
                    <><motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                      className="w-4 h-4 rounded-full border-2 border-white border-t-transparent" />
                      Sending...</>
                  ) : (
                    <><HiMail /> Send Inquiry</>
                  )}
                </motion.button>
              </motion.form>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  )
}
