"use client"

import { Button } from "@/components/ui/button"
import { ParticleBackground } from "@/components/particle-background"
import { motion } from "framer-motion"
import { AlertCircle, RefreshCw, Home, Bug } from "lucide-react"
import Link from "next/link"
import { useEffect, useState } from "react"

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  const [windowWidth, setWindowWidth] = useState(0)

  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth)
    }

    setWindowWidth(window.innerWidth)
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  useEffect(() => {
    // Log the error to an error reporting service
    console.error(error)
  }, [error])

  const getParticleDensity = () => {
    if (windowWidth < 768) return "low"
    if (windowWidth < 1280) return "medium"
    return "high"
  }

  return (
    <div className="flex items-center justify-center min-h-screen w-full relative">
      {/* Multiple particle layers for depth */}
      <div className="particle-container fixed inset-0 z-0">
        <ParticleBackground variant="starfield" density={getParticleDensity()} />
      </div>
      <div className="particle-container fixed inset-0 z-0 opacity-60">
        <ParticleBackground variant="cosmic" density="low" />
      </div>
      <div className="particle-container fixed inset-0 z-0 opacity-30">
        <ParticleBackground variant="nebula" density="low" />
      </div>
      <div className="particle-container fixed inset-0 z-0 opacity-20">
        <ParticleBackground variant="hero" density="low" />
      </div>

      <div className="container px-4 z-10">
        <motion.div
          className="max-w-lg mx-auto text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <motion.div
            className="mb-8 flex justify-center"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <div className="relative">
              <div className="h-32 w-32 rounded-full bg-red-500/20 border-2 border-red-500/30 flex items-center justify-center backdrop-blur-sm">
                <AlertCircle className="h-16 w-16 text-red-400 animate-pulse" />
              </div>
              <motion.div
                className="absolute -top-3 -right-3 h-12 w-12 rounded-full bg-cosmic-cyan/20 border border-cosmic-cyan/30 flex items-center justify-center backdrop-blur-sm"
                animate={{
                  rotate: [0, 360],
                }}
                transition={{
                  duration: 4,
                  repeat: Number.POSITIVE_INFINITY,
                  ease: "linear",
                }}
              >
                <Bug className="h-6 w-6 text-cosmic-cyan" />
              </motion.div>
            </div>
          </motion.div>

          <motion.h1
            className="text-4xl md:text-5xl font-bold cosmic-text mb-4"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            System Malfunction
          </motion.h1>

          <motion.h2
            className="text-2xl md:text-3xl font-semibold text-space-white mb-4"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            Cosmic Interference Detected
          </motion.h2>

          <motion.p
            className="text-lg text-space-white/70 mb-8 leading-relaxed"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            Our cosmic navigation systems have encountered an unexpected anomaly. The stellar winds may have disrupted
            our quantum processors.
          </motion.p>

          {error.digest && (
            <motion.div
              className="mb-6 p-4 bg-black/30 border border-red-500/30 rounded-lg"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.6 }}
            >
              <p className="text-sm text-red-400 font-mono">Error ID: {error.digest}</p>
            </motion.div>
          )}

          <motion.div
            className="flex flex-col sm:flex-row gap-4 justify-center mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.7 }}
          >
            <Button onClick={reset} className="cosmic-button gap-2 text-space-white px-6 py-3">
              <RefreshCw className="h-5 w-5" />
              Recalibrate Systems
            </Button>

            <Button
              asChild
              variant="outline"
              className="gap-2 bg-black/30 border-white/20 text-space-white hover:bg-cosmic-purple/20 hover:border-cosmic-purple px-6 py-3"
            >
              <Link href="/">
                <Home className="h-5 w-5" />
                Return to Home Base
              </Link>
            </Button>
          </motion.div>

          <motion.div
            className="text-sm text-space-white/50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.8 }}
          >
            <p>If the problem persists, our cosmic engineers have been notified.</p>
          </motion.div>
        </motion.div>
      </div>
    </div>
  )
}
