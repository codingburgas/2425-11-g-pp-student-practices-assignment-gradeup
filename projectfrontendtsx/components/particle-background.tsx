"use client"

import { useCallback, useEffect, useState } from "react"
import Particles from "react-particles"
import type { Container, Engine } from "tsparticles-engine"
import { loadSlim } from "tsparticles-slim"
import { useTheme } from "next-themes"

interface ParticleBackgroundProps {
  id?: string
  className?: string
  variant?: "default" | "auth" | "hero" | "dense" | "cosmic" | "nebula" | "starfield"
  density?: "low" | "medium" | "high" | "ultra"
}

export function ParticleBackground({
  id = "tsparticles",
  className,
  variant = "default",
  density = "medium",
}: ParticleBackgroundProps) {
  const [mounted, setMounted] = useState(false)
  const { theme } = useTheme()
  const isDarkTheme = theme === "dark"

  useEffect(() => {
    setMounted(true)
  }, [])

  const particlesInit = useCallback(async (engine: Engine) => {
    await loadSlim(engine)
  }, [])

  const particlesLoaded = useCallback(async (container: Container | undefined) => {
    // Container loaded
  }, [])

  if (!mounted) return null

  // Determine particle count based on density
  const getParticleCount = () => {
    const baseCounts = {
      low: { default: 30, hero: 50, cosmic: 80, nebula: 60, starfield: 100 },
      medium: { default: 60, hero: 100, cosmic: 150, nebula: 120, starfield: 200 },
      high: { default: 120, hero: 180, cosmic: 250, nebula: 200, starfield: 300 },
      ultra: { default: 200, hero: 300, cosmic: 400, nebula: 350, starfield: 500 },
    }

    return baseCounts[density][variant] || baseCounts[density].default
  }

  // Space-themed color palette
  const getSpaceColors = () => {
    return {
      primary: isDarkTheme ? "#8B5CF6" : "#7C3AED",
      secondary: isDarkTheme ? "#06B6D4" : "#0891B2",
      accent: isDarkTheme ? "#EC4899" : "#DB2777",
      cosmic: isDarkTheme ? "#F59E0B" : "#D97706",
      nebula: isDarkTheme ? "#8B5CF6" : "#7C3AED",
      star: isDarkTheme ? "#F8FAFC" : "#1E293B",
    }
  }

  const getConfig = () => {
    const colors = getSpaceColors()
    const particleCount = getParticleCount()

    switch (variant) {
      case "cosmic":
        return {
          fullScreen: { enable: false },
          fpsLimit: 120,
          particles: {
            number: {
              value: particleCount,
              density: { enable: true, value_area: 800 },
            },
            color: {
              value: [colors.primary, colors.secondary, colors.accent, colors.cosmic],
            },
            shape: {
              type: ["circle", "triangle", "polygon", "star"],
              polygon: { sides: 6 },
              star: { sides: 5 },
            },
            opacity: {
              value: { min: 0.1, max: 0.8 },
              random: true,
              anim: {
                enable: true,
                speed: 1,
                opacity_min: 0.1,
                sync: false,
              },
            },
            size: {
              value: { min: 1, max: 8 },
              random: true,
              anim: {
                enable: true,
                speed: 3,
                size_min: 0.3,
                sync: false,
              },
            },
            move: {
              enable: true,
              speed: { min: 0.5, max: 2 },
              direction: "none",
              random: true,
              straight: false,
              out_mode: "out",
              bounce: false,
              attract: {
                enable: true,
                rotateX: 600,
                rotateY: 1200,
              },
            },
            links: {
              enable: true,
              distance: 150,
              color: colors.primary,
              opacity: 0.3,
              width: 1,
              triangles: {
                enable: true,
                color: colors.accent,
                opacity: 0.1,
              },
            },
            twinkle: {
              particles: {
                enable: true,
                frequency: 0.05,
                opacity: 1,
              },
            },
          },
          interactivity: {
            detect_on: "canvas",
            events: {
              onhover: {
                enable: true,
                mode: ["grab", "bubble"],
              },
              onclick: {
                enable: true,
                mode: "push",
              },
              resize: true,
            },
            modes: {
              grab: {
                distance: 200,
                links: { opacity: 0.6 },
              },
              bubble: {
                distance: 200,
                size: 12,
                duration: 2,
                opacity: 0.8,
                speed: 3,
              },
              push: { particles_nb: 8 },
            },
          },
          retina_detect: true,
          background: { color: "transparent" },
        }

      case "nebula":
        return {
          fullScreen: { enable: false },
          fpsLimit: 60,
          particles: {
            number: {
              value: particleCount,
              density: { enable: true, value_area: 1000 },
            },
            color: {
              value: [colors.nebula, colors.accent, colors.secondary],
            },
            shape: {
              type: ["circle", "edge"],
            },
            opacity: {
              value: { min: 0.1, max: 0.6 },
              random: true,
              anim: {
                enable: true,
                speed: 0.5,
                opacity_min: 0.1,
                sync: false,
              },
            },
            size: {
              value: { min: 5, max: 20 },
              random: true,
              anim: {
                enable: true,
                speed: 1,
                size_min: 2,
                sync: false,
              },
            },
            move: {
              enable: true,
              speed: 0.3,
              direction: "none",
              random: true,
              straight: false,
              out_mode: "out",
              bounce: false,
            },
            links: {
              enable: false,
            },
          },
          interactivity: {
            detect_on: "canvas",
            events: {
              onhover: {
                enable: true,
                mode: "repulse",
              },
              onclick: {
                enable: true,
                mode: "bubble",
              },
              resize: true,
            },
            modes: {
              repulse: {
                distance: 100,
                duration: 0.4,
              },
              bubble: {
                distance: 150,
                size: 25,
                duration: 2,
                opacity: 0.8,
                speed: 3,
              },
            },
          },
          retina_detect: true,
          background: { color: "transparent" },
        }

      case "starfield":
        return {
          fullScreen: { enable: false },
          fpsLimit: 120,
          particles: {
            number: {
              value: particleCount,
              density: { enable: true, value_area: 600 },
            },
            color: {
              value: [colors.star, colors.primary, colors.secondary],
            },
            shape: {
              type: ["circle", "star"],
              star: { sides: 4 },
            },
            opacity: {
              value: { min: 0.2, max: 1 },
              random: true,
              anim: {
                enable: true,
                speed: 2,
                opacity_min: 0.1,
                sync: false,
              },
            },
            size: {
              value: { min: 0.5, max: 3 },
              random: true,
              anim: {
                enable: true,
                speed: 4,
                size_min: 0.1,
                sync: false,
              },
            },
            move: {
              enable: true,
              speed: { min: 0.1, max: 0.5 },
              direction: "none",
              random: true,
              straight: false,
              out_mode: "out",
              bounce: false,
            },
            links: {
              enable: false,
            },
            twinkle: {
              particles: {
                enable: true,
                frequency: 0.1,
                opacity: 1,
              },
            },
          },
          interactivity: {
            detect_on: "canvas",
            events: {
              onhover: {
                enable: true,
                mode: "grab",
              },
              onclick: {
                enable: true,
                mode: "push",
              },
              resize: true,
            },
            modes: {
              grab: {
                distance: 100,
                links: { opacity: 0.5 },
              },
              push: { particles_nb: 6 },
            },
          },
          retina_detect: true,
          background: { color: "transparent" },
        }

      case "hero":
        return {
          fullScreen: { enable: false },
          fpsLimit: 120,
          particles: {
            number: {
              value: particleCount,
              density: { enable: true, value_area: 800 },
            },
            color: {
              value: [colors.star, colors.primary, colors.secondary],
            },
            shape: {
              type: ["circle", "triangle", "star"],
              star: { sides: 5 },
            },
            opacity: {
              value: { min: 0.2, max: 0.8 },
              random: true,
              anim: {
                enable: true,
                speed: 1.5,
                opacity_min: 0.1,
                sync: false,
              },
            },
            size: {
              value: { min: 1, max: 6 },
              random: true,
              anim: {
                enable: true,
                speed: 2,
                size_min: 0.3,
                sync: false,
              },
            },
            move: {
              enable: true,
              speed: 1.5,
              direction: "none",
              random: true,
              straight: false,
              out_mode: "out",
              bounce: false,
            },
            links: {
              enable: true,
              distance: 150,
              color: colors.star,
              opacity: 0.3,
              width: 1,
            },
            twinkle: {
              particles: {
                enable: true,
                frequency: 0.05,
                opacity: 1,
              },
            },
          },
          interactivity: {
            detect_on: "canvas",
            events: {
              onhover: {
                enable: true,
                mode: "grab",
              },
              onclick: {
                enable: true,
                mode: "repulse",
              },
              resize: true,
            },
            modes: {
              grab: {
                distance: 180,
                links: { opacity: 0.5 },
              },
              repulse: {
                distance: 200,
                duration: 0.4,
              },
            },
          },
          retina_detect: true,
          background: { color: "transparent" },
        }

      case "auth":
        return {
          fullScreen: { enable: false },
          fpsLimit: 120,
          particles: {
            number: {
              value: particleCount,
              density: { enable: true, value_area: 800 },
            },
            color: {
              value: [colors.primary, colors.secondary, colors.accent],
            },
            shape: {
              type: ["circle", "triangle"],
            },
            opacity: {
              value: { min: 0.3, max: 0.7 },
              random: true,
              anim: {
                enable: true,
                speed: 1,
                opacity_min: 0.1,
                sync: false,
              },
            },
            size: {
              value: { min: 2, max: 6 },
              random: true,
              anim: {
                enable: true,
                speed: 2,
                size_min: 0.5,
                sync: false,
              },
            },
            move: {
              enable: true,
              speed: 1.5,
              direction: "none",
              random: true,
              straight: false,
              out_mode: "out",
              bounce: false,
            },
            links: {
              enable: true,
              distance: 120,
              color: colors.primary,
              opacity: 0.4,
              width: 1,
            },
          },
          interactivity: {
            detect_on: "canvas",
            events: {
              onhover: {
                enable: true,
                mode: "bubble",
              },
              onclick: {
                enable: true,
                mode: "push",
              },
              resize: true,
            },
            modes: {
              bubble: {
                distance: 150,
                size: 8,
                duration: 2,
                opacity: 0.8,
                speed: 3,
              },
              push: { particles_nb: 4 },
            },
          },
          retina_detect: true,
          background: { color: "transparent" },
        }

      default:
        return {
          fullScreen: { enable: false },
          fpsLimit: 120,
          particles: {
            number: {
              value: particleCount,
              density: { enable: true, value_area: 800 },
            },
            color: {
              value: [colors.primary, colors.secondary],
            },
            shape: {
              type: "circle",
            },
            opacity: {
              value: 0.4,
              random: true,
              anim: {
                enable: true,
                speed: 1,
                opacity_min: 0.1,
                sync: false,
              },
            },
            size: {
              value: { min: 1, max: 4 },
              random: true,
              anim: {
                enable: true,
                speed: 2,
                size_min: 0.3,
                sync: false,
              },
            },
            move: {
              enable: true,
              speed: 1,
              direction: "none",
              random: true,
              straight: false,
              out_mode: "out",
              bounce: false,
            },
          },
          interactivity: {
            detect_on: "canvas",
            events: {
              onhover: {
                enable: true,
                mode: "repulse",
              },
              onclick: {
                enable: true,
                mode: "push",
              },
              resize: true,
            },
            modes: {
              repulse: {
                distance: 100,
                duration: 0.4,
              },
              push: { particles_nb: 4 },
            },
          },
          retina_detect: true,
          background: { color: "transparent" },
        }
    }
  }

  return (
    <Particles
      id={id}
      className={className}
      init={particlesInit}
      loaded={particlesLoaded}
      options={getConfig()}
      style={{ width: "100%", height: "100%" }}
    />
  )
}
