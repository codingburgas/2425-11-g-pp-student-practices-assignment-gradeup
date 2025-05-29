"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/components/ui/use-toast"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { z } from "zod"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ParticleBackground } from "@/components/particle-background"
import { motion } from "framer-motion"
import { User, Mail, Lock, UserCheck, ArrowRight, Eye, EyeOff, Rocket } from "lucide-react"

const registerSchema = z
  .object({
    name: z.string().min(2, { message: "Name must be at least 2 characters" }),
    email: z.string().email({ message: "Please enter a valid email address" }),
    password: z.string().min(6, { message: "Password must be at least 6 characters" }),
    confirmPassword: z.string(),
    role: z.enum(["student", "admin"]),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords do not match",
    path: ["confirmPassword"],
  })

export default function RegisterPage() {
  const { toast } = useToast()
  const router = useRouter()
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
    role: "student",
  })
  const [errors, setErrors] = useState({})
  const [isLoading, setIsLoading] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [windowWidth, setWindowWidth] = useState(0)

  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth)
    }

    setWindowWidth(window.innerWidth)
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  const handleChange = (e) => {
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

  const handleRoleChange = (value) => {
    setFormData((prev) => ({ ...prev, role: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      registerSchema.parse(formData)
      await new Promise((resolve) => setTimeout(resolve, 1000))

      toast({
        title: "Registration successful",
        description: "Welcome to the cosmic journey with GradeUP!",
      })

      router.push("/login")
    } catch (error) {
      if (error instanceof z.ZodError) {
        const newErrors = {}
        error.errors.forEach((err) => {
          if (err.path[0]) {
            newErrors[err.path[0]] = err.message
          }
        })
        setErrors(newErrors)
      } else {
        toast({
          variant: "destructive",
          title: "Registration failed",
          description: "There was an error creating your account. Please try again.",
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
        <Card className="cosmic-form border-2 border-white/20 shadow-2xl">
          <CardHeader className="space-y-1 text-center">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="flex items-center justify-center mb-4"
            >
              <div className="relative">
                <Rocket className="h-8 w-8 text-cosmic-purple animate-pulse-glow" />
                <div className="absolute inset-0 h-8 w-8 text-cosmic-cyan animate-pulse-glow opacity-50"></div>
              </div>
            </motion.div>
            <CardTitle className="text-2xl sm:text-3xl font-bold cosmic-text">Join the Cosmos</CardTitle>
            <CardDescription className="text-space-white/70">
              Begin your journey through the universe of education
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-5">
              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.3 }}
              >
                <Label htmlFor="name" className="text-space-white/90">
                  Full Name
                </Label>
                <div className="relative group">
                  <User className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cosmic-purple transition-colors group-focus-within:text-cosmic-cyan" />
                  <Input
                    id="name"
                    name="name"
                    placeholder="Cosmic Explorer"
                    value={formData.name}
                    onChange={handleChange}
                    className={`pl-10 bg-black/30 border-white/20 text-space-white placeholder:text-space-white/50 focus:border-cosmic-purple focus:ring-cosmic-purple/50 transition-all duration-300 ${
                      errors.name ? "border-red-500 focus:border-red-500 focus:ring-red-500/50" : ""
                    }`}
                  />
                </div>
                {errors.name && (
                  <motion.p
                    className="text-sm text-red-400"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                  >
                    {errors.name}
                  </motion.p>
                )}
              </motion.div>

              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.4 }}
              >
                <Label htmlFor="email" className="text-space-white/90">
                  Email
                </Label>
                <div className="relative group">
                  <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cosmic-purple transition-colors group-focus-within:text-cosmic-cyan" />
                  <Input
                    id="email"
                    name="email"
                    type="email"
                    placeholder="explorer@cosmos.com"
                    value={formData.email}
                    onChange={handleChange}
                    className={`pl-10 bg-black/30 border-white/20 text-space-white placeholder:text-space-white/50 focus:border-cosmic-purple focus:ring-cosmic-purple/50 transition-all duration-300 ${
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
                transition={{ duration: 0.4, delay: 0.5 }}
              >
                <Label htmlFor="password" className="text-space-white/90">
                  Password
                </Label>
                <div className="relative group">
                  <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cosmic-purple transition-colors group-focus-within:text-cosmic-cyan" />
                  <Input
                    id="password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    value={formData.password}
                    onChange={handleChange}
                    className={`pl-10 pr-10 bg-black/30 border-white/20 text-space-white placeholder:text-space-white/50 focus:border-cosmic-purple focus:ring-cosmic-purple/50 transition-all duration-300 ${
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
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.6 }}
              >
                <Label htmlFor="confirmPassword" className="text-space-white/90">
                  Confirm Password
                </Label>
                <div className="relative group">
                  <UserCheck className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cosmic-purple transition-colors group-focus-within:text-cosmic-cyan" />
                  <Input
                    id="confirmPassword"
                    name="confirmPassword"
                    type={showConfirmPassword ? "text" : "password"}
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    className={`pl-10 pr-10 bg-black/30 border-white/20 text-space-white placeholder:text-space-white/50 focus:border-cosmic-purple focus:ring-cosmic-purple/50 transition-all duration-300 ${
                      errors.confirmPassword ? "border-red-500 focus:border-red-500 focus:ring-red-500/50" : ""
                    }`}
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-cosmic-purple hover:text-cosmic-cyan transition-colors duration-200"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    tabIndex={-1}
                  >
                    {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
                {errors.confirmPassword && (
                  <motion.p
                    className="text-sm text-red-400"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                  >
                    {errors.confirmPassword}
                  </motion.p>
                )}
              </motion.div>

              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.7 }}
              >
                <Label htmlFor="role" className="text-space-white/90">
                  Role
                </Label>
                <Select value={formData.role} onValueChange={handleRoleChange}>
                  <SelectTrigger className="bg-black/30 border-white/20 text-space-white focus:border-cosmic-purple focus:ring-cosmic-purple/50 transition-all duration-300">
                    <SelectValue placeholder="Select your cosmic role" />
                  </SelectTrigger>
                  <SelectContent className="bg-black/90 border-white/20 text-space-white">
                    <SelectItem value="student" className="focus:bg-cosmic-purple/20">
                      Student Explorer
                    </SelectItem>
                    <SelectItem value="admin" className="focus:bg-cosmic-purple/20">
                      Cosmic Administrator
                    </SelectItem>
                  </SelectContent>
                </Select>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.8 }}
              >
                <Button
                  type="submit"
                  className="w-full cosmic-button text-space-white font-medium py-3 transition-all duration-300"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <div className="flex items-center">
                      <div className="animate-spin -ml-1 mr-3 h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                      Launching into cosmos...
                    </div>
                  ) : (
                    <div className="flex items-center">
                      Create account
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
              transition={{ duration: 0.4, delay: 0.9 }}
            >
              <span className="text-space-white/70">Already exploring the cosmos? </span>
              <Link
                href="/login"
                className="text-cosmic-cyan hover:text-cosmic-purple font-medium transition-colors duration-200"
              >
                Return to base
              </Link>
            </motion.div>
          </CardFooter>
        </Card>
      </motion.div>
    </div>
  )
}
