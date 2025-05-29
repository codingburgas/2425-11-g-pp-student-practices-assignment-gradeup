"use client"

import { Button } from "@/components/ui/button"
import { ParticleBackground } from "@/components/particle-background"
import { motion } from "framer-motion"
import { Zap, RefreshCw, AlertTriangle } from "lucide-react"
import { useEffect, useState } from "react"

export default function GlobalError({
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
    <html>
      <body>
        <div className="flex items-center justify-center min-h-screen w-full relative bg-black">
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
                  <div className="h-32 w-32 rounded-full bg-red-600/20 border-2 border-red-600/30 flex items-center justify-center backdrop-blur-sm">
                    <AlertTriangle className="h-16 w-16 text-red-500 animate-pulse" />
                  </div>
                  <motion.div
                    className="absolute -top-3 -right-3 h-12 w-12 rounded-full bg-yellow-500/20 border border-yellow-500/30 flex items-center justify-center backdrop-blur-sm"
                    animate={{
                      scale: [1, 1.3, 1],
                      rotate: [0, 180, 360],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Number.POSITIVE_INFINITY,
                    }}
                  >
                    <Zap className="h-6 w-6 text-yellow-400" />
                  </motion.div>
                </div>
              </motion.div>

              <motion.h1
                className="text-4xl md:text-5xl font-bold text-white mb-4"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                style={{
                  background: "linear-gradient(45deg, #8B5CF6, #06B6D4)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundClip: "text",
                }}
              >
                Critical System Failure
              </motion.h1>

              <motion.h2
                className="text-2xl md:text-3xl font-semibold text-white mb-4"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
              >
                Cosmic Core Meltdown
              </motion.h2>

              <motion.p
                className="text-lg text-white/70 mb-8 leading-relaxed"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.5 }}
              >
                A catastrophic failure has occurred in our cosmic infrastructure. The quantum matrix has destabilized
                and requires immediate attention.
              </motion.p>

              {error.digest && (
                <motion.div
                  className="mb-6 p-4 bg-red-900/20 border border-red-500/30 rounded-lg"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.6 }}
                >
                  <p className="text-sm text-red-400 font-mono">Critical Error ID: {error.digest}</p>
                </motion.div>
              )}

              <motion.div
                className="flex flex-col sm:flex-row gap-4 justify-center mb-8"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.7 }}
              >
                <Button
                  onClick={reset}
                  className="gap-2 text-white px-6 py-3 bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-700 hover:to-cyan-700"
                >
                  <RefreshCw className="h-5 w-5" />
                  Emergency Restart
                </Button>

                <Button
                  onClick={() => (window.location.href = "/")}
                  variant="outline"
                  className="gap-2 bg-black/30 border-white/20 text-white hover:bg-purple-600/20 hover:border-purple-500 px-6 py-3"
                >
                  <Zap className="h-5 w-5" />
                  Force Reload
                </Button>
              </motion.div>

              <motion.div
                className="text-sm text-white/50"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.8 }}
              >
                <p>Emergency protocols have been activated. Our cosmic engineers are responding.</p>
              </motion.div>
            </motion.div>
          </div>
        </div>
      </body>
    </html>
  )
}
