"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/components/ui/use-toast"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { z } from "zod"
import { ParticleBackground } from "@/components/particle-background"
import { motion } from "framer-motion"
import { Mail, Lock, ArrowRight, Eye, EyeOff, Sparkles } from "lucide-react"

const loginSchema = z.object({
  email: z.string().email({ message: "Please enter a valid email address" }),
  password: z.string().min(6, { message: "Password must be at least 6 characters" }),
})

export default function LoginPage() {
  const { toast } = useToast()
  const router = useRouter()
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [isLoading, setIsLoading] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [windowWidth, setWindowWidth] = useState(0)

  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth)
    }

    setWindowWidth(window.innerWidth)
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))

    if (errors[name]) {
      setErrors((prev) => {
        const newErrors = { ...prev }
        delete newErrors[name]
        return newErrors
      })
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      loginSchema.parse(formData)
      await new Promise((resolve) => setTimeout(resolve, 1000))

      toast({
        title: "Login successful",
        description: "Welcome back to GradeUP!",
      })

      router.push("/")
    } catch (error) {
      if (error instanceof z.ZodError) {
        const newErrors: Record<string, string> = {}
        error.errors.forEach((err) => {
          if (err.path[0]) {
            newErrors[err.path[0] as string] = err.message
          }
        })
        setErrors(newErrors)
      } else {
        toast({
          variant: "destructive",
          title: "Login failed",
          description: "Please check your credentials and try again.",
        })
      }
    } finally {
      setIsLoading(false)
    }
  }

  const getParticleDensity = () => {
    if (windowWidth < 768) return "low"
    if (windowWidth < 1280) return "medium"
    return "high"
  }

  return (
    <div className="flex items-center justify-center min-h-[calc(100vh-4rem)] w-full p-4">
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

      <motion.div
        className="w-full max-w-md z-10"
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        <Card className="cosmic-form border-2 border-cosmic-purple/30 shadow-2xl">
          <CardHeader className="space-y-1 text-center">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="flex items-center justify-center mb-4"
            >
              <div className="relative">
                <Sparkles className="h-8 w-8 text-cosmic-purple animate-pulse-glow" />
                <div className="absolute inset-0 h-8 w-8 text-cosmic-cyan animate-pulse-glow opacity-50"></div>
              </div>
            </motion.div>
            <CardTitle className="text-2xl sm:text-3xl font-bold cosmic-text">Welcome Back</CardTitle>
            <CardDescription className="text-muted-foreground">
              Enter your credentials to access your cosmic journey
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.3 }}
              >
                <Label htmlFor="email" className="text-foreground">
                  Email
                </Label>
                <div className="relative group">
                  <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cosmic-purple transition-colors group-focus-within:text-cosmic-cyan" />
                  <Input
                    id="email"
                    name="email"
                    type="email"
                    placeholder="your.email@cosmos.com"
                    value={formData.email}
                    onChange={handleChange}
                    className={`pl-10 cosmic-input transition-all duration-300 ${
                      errors.email ? "border-red-500 focus:border-red-500 focus:ring-red-500/50" : ""
                    }`}
                  />
                </div>
                {errors.email && (
                  <motion.p
                    className="text-sm text-red-400"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                  >
                    {errors.email}
                  </motion.p>
                )}
              </motion.div>

              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.4 }}
              >
                <div className="flex items-center justify-between">
                  <Label htmlFor="password" className="text-foreground">
                    Password
                  </Label>
                  <Link
                    href="/forgot-password"
                    className="text-xs text-cosmic-cyan hover:text-cosmic-purple transition-colors duration-200"
                  >
                    Forgot password?
                  </Link>
                </div>
                <div className="relative group">
                  <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cosmic-purple transition-colors group-focus-within:text-cosmic-cyan" />
                  <Input
                    id="password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    value={formData.password}
                    onChange={handleChange}
                    className={`pl-10 pr-10 cosmic-input transition-all duration-300 ${
                      errors.password ? "border-red-500 focus:border-red-500 focus:ring-red-500/50" : ""
                    }`}
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-cosmic-purple hover:text-cosmic-cyan transition-colors duration-200"
                    onClick={() => setShowPassword(!showPassword)}
                    tabIndex={-1}
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
                {errors.password && (
                  <motion.p
                    className="text-sm text-red-400"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                  >
                    {errors.password}
                  </motion.p>
                )}
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.5 }}
              >
                <Button
                  type="submit"
                  className="w-full cosmic-button font-medium py-3 transition-all duration-300"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <div className="flex items-center">
                      <div className="animate-spin -ml-1 mr-3 h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                      Entering the cosmos...
                    </div>
                  ) : (
                    <div className="flex items-center">
                      Login
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </div>
                  )}
                </Button>
              </motion.div>
            </form>
          </CardContent>
          <CardFooter className="flex flex-col space-y-4">
            <motion.div
              className="text-center text-sm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, delay: 0.6 }}
            >
              <span className="text-muted-foreground">Don&apos;t have an account? </span>
              <Link
                href="/register"
                className="text-cosmic-cyan hover:text-cosmic-purple font-medium transition-colors duration-200"
              >
                Join the cosmos
              </Link>
            </motion.div>
          </CardFooter>
        </Card>
      </motion.div>
    </div>
  )
}
