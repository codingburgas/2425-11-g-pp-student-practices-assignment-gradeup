"use client"

import { Button } from "@/components/ui/button"
import { ParticleBackground } from "@/components/particle-background"
import { motion } from "framer-motion"
import { AlertTriangle, Home, Search, ArrowLeft, Telescope } from "lucide-react"
import Link from "next/link"
import { useEffect, useState } from "react"

export default function NotFound() {
  const [windowWidth, setWindowWidth] = useState(0)

  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth)
    }

    setWindowWidth(window.innerWidth)
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

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
              <div className="h-32 w-32 rounded-full bg-cosmic-purple/20 border-2 border-cosmic-purple/30 flex items-center justify-center backdrop-blur-sm">
                <AlertTriangle className="h-16 w-16 text-cosmic-purple animate-pulse" />
              </div>
              <motion.div
                className="absolute -top-3 -right-3 h-12 w-12 rounded-full bg-cosmic-cyan/20 border border-cosmic-cyan/30 flex items-center justify-center backdrop-blur-sm"
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.7, 1, 0.7],
                }}
                transition={{
                  duration: 3,
                  repeat: Number.POSITIVE_INFINITY,
                  repeatType: "reverse",
                }}
              >
                <span className="text-cosmic-cyan text-2xl font-bold">?</span>
              </motion.div>
              <motion.div
                className="absolute -bottom-2 -left-2 h-8 w-8 rounded-full bg-cosmic-purple/30 flex items-center justify-center"
                animate={{
                  rotate: [0, 360],
                }}
                transition={{
                  duration: 8,
                  repeat: Number.POSITIVE_INFINITY,
                  ease: "linear",
                }}
              >
                <Telescope className="h-4 w-4 text-cosmic-purple" />
              </motion.div>
            </div>
          </motion.div>

          <motion.h1
            className="text-6xl md:text-7xl font-bold cosmic-text mb-4"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            404
          </motion.h1>

          <motion.h2
            className="text-3xl md:text-4xl font-semibold text-foreground mb-4"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            Lost in the Cosmic Void
          </motion.h2>

          <motion.p
            className="text-lg text-muted-foreground mb-8 leading-relaxed"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            The cosmic coordinates you're seeking have drifted beyond our observable universe. This stellar destination
            may have collapsed into a black hole or exists in a parallel dimension.
          </motion.p>

          <motion.div
            className="flex flex-col sm:flex-row gap-4 justify-center mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <Button asChild className="cosmic-button gap-2 px-6 py-3">
              <Link href="/">
                <Home className="h-5 w-5" />
                Return to Home Base
              </Link>
            </Button>

            <Button
              asChild
              variant="outline"
              className="gap-2 border-cosmic-purple/30 text-cosmic-purple hover:bg-cosmic-purple/10 hover:border-cosmic-purple px-6 py-3"
            >
              <Link href="/universities">
                <Search className="h-5 w-5" />
                Explore Universities
              </Link>
            </Button>
          </motion.div>

          <motion.div
            className="text-sm text-muted-foreground"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.7 }}
          >
            <Button
              variant="link"
              className="text-cosmic-cyan hover:text-cosmic-purple p-0 h-auto"
              onClick={() => window.history.back()}
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Navigate back to previous coordinates
            </Button>
          </motion.div>
        </motion.div>
      </div>
    </div>
  )
}
