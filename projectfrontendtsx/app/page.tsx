"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import {
  GraduationCap,
  School,
  BookOpen,
  Users,
  ArrowRight,
  CheckCircle,
  Star,
  Award,
  TrendingUp,
  Clock,
  BarChart,
  Target,
  Compass,
  Lightbulb,
  Heart,
  ThumbsUp,
  ClipboardList,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"
import Link from "next/link"
import Image from "next/image"
import { ParticleBackground } from "@/components/particle-background"
import { motion, AnimatePresence } from "framer-motion"
import { useInView } from "react-intersection-observer"
import { useState, useEffect } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
}

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
    },
  },
}

const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
    },
  },
  hover: {
    scale: 1.05,
    boxShadow: "0 10px 30px rgba(0, 0, 0, 0.1)",
    transition: {
      duration: 0.3,
    },
  },
}

// Animated counter component
const AnimatedCounter = ({ value, duration = 2, className = "" }) => {
  const [count, setCount] = useState(0)
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 })

  useEffect(() => {
    if (!inView) return

    let start = 0
    const end = Number.parseInt(value.toString().replace(/,/g, ""))
    const incrementTime = (duration * 1000) / end

    const timer = setInterval(() => {
      start += 1
      setCount(start)
      if (start >= end) clearInterval(timer)
    }, incrementTime)

    return () => {
      clearInterval(timer)
    }
  }, [value, duration, inView])

  return (
    <span ref={ref} className={`counter-animation ${className}`}>
      {count.toLocaleString()}
    </span>
  )
}

// Testimonial data
const testimonials = [
  {
    id: 1,
    name: "Maria Petrova",
    role: "Computer Science Student",
    university: "Sofia University",
    quote:
      "GradeUP helped me find the perfect Computer Science program. The personalized recommendations were spot on!",
    avatar: "/placeholder.svg?height=100&width=100",
  },
  {
    id: 2,
    name: "Ivan Dimitrov",
    role: "Business Administration Student",
    university: "American University in Bulgaria",
    quote:
      "The survey was comprehensive and the university matches were exactly what I was looking for. Highly recommend!",
    avatar: "/placeholder.svg?height=100&width=100",
  },
  {
    id: 3,
    name: "Elena Ivanova",
    role: "Medicine Student",
    university: "Medical University of Sofia",
    quote: "As someone who was unsure about my path, GradeUP provided clarity and direction for my academic journey.",
    avatar: "/placeholder.svg?height=100&width=100",
  },
]

// FAQ data
const faqs = [
  {
    question: "How does GradeUP generate university recommendations?",
    answer:
      "GradeUP uses a sophisticated algorithm that analyzes your academic interests, skills, learning style, and career goals from your survey responses. We match these with university programs that align with your profile, considering factors like program content, teaching methods, and career outcomes.",
  },
  {
    question: "Is GradeUP only for high school students?",
    answer:
      "While GradeUP is particularly helpful for students finishing 7th grade and high school students planning their university education, it's also valuable for anyone considering a career change or further education. Our platform provides insights for learners at various stages of their educational journey.",
  },
  {
    question: "How accurate are the university recommendations?",
    answer:
      "Our recommendations are highly accurate based on the information you provide. The more detailed your survey responses, the more tailored our recommendations will be. We continuously refine our algorithm based on user feedback and educational outcomes to improve accuracy.",
  },
  {
    question: "Can I save and compare different university recommendations?",
    answer:
      "Yes! Once you create an account, you can save your survey results and recommendations. Our comparison tool allows you to evaluate different universities and programs side by side, considering factors like location, program content, tuition fees, and career prospects.",
  },
  {
    question: "Does GradeUP offer information about scholarships and financial aid?",
    answer:
      "Yes, we provide information about available scholarships, grants, and financial aid options for each recommended university. We understand that financial considerations are an important part of the decision-making process for many students.",
  },
]

// Timeline data
const timeline = [
  {
    title: "Take the Survey",
    description:
      "Answer questions about your interests, skills, and academic preferences to help us understand your unique profile.",
    icon: ClipboardList,
  },
  {
    title: "Get Personalized Recommendations",
    description:
      "Our algorithm analyzes your responses to generate tailored university and program suggestions that match your profile.",
    icon: Target,
  },
  {
    title: "Explore Universities",
    description:
      "Browse detailed information about recommended universities, including programs, facilities, and admission requirements.",
    icon: Compass,
  },
  {
    title: "Compare Options",
    description: "Use our comparison tools to evaluate different universities and programs side by side.",
    icon: BarChart,
  },
  {
    title: "Make Informed Decisions",
    description:
      "Choose the best path for your academic and career goals with confidence, backed by data and insights.",
    icon: Lightbulb,
  },
]

export default function Home() {
  const [heroRef, heroInView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const [featuresRef, featuresInView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const [statsRef, statsInView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const [timelineRef, timelineInView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const [testimonialsRef, testimonialsInView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const [faqRef, faqInView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const [ctaRef, ctaInView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const [activeTestimonial, setActiveTestimonial] = useState(0)
  const [windowWidth, setWindowWidth] = useState(0)

  // Track window width for responsive adjustments
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth)
    }

    // Set initial width
    setWindowWidth(window.innerWidth)

    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  // Auto-rotate testimonials
  useEffect(() => {
    if (testimonialsInView) {
      const interval = setInterval(() => {
        setActiveTestimonial((prev) => (prev + 1) % testimonials.length)
      }, 5000)
      return () => clearInterval(interval)
    }
  }, [testimonialsInView])

  // Manual testimonial navigation
  const nextTestimonial = () => {
    setActiveTestimonial((prev) => (prev + 1) % testimonials.length)
  }

  const prevTestimonial = () => {
    setActiveTestimonial((prev) => (prev - 1 + testimonials.length) % testimonials.length)
  }

  // Determine particle density based on screen size
  const getParticleDensity = () => {
    if (windowWidth < 768) return "low"
    if (windowWidth < 1280) return "medium"
    return "high"
  }

  return (
    <div className="container mx-auto space-y-16 md:space-y-20 lg:space-y-24 py-6">
      {/* Hero Section */}
      <motion.section
        ref={heroRef}
        initial="hidden"
        animate={heroInView ? "visible" : "hidden"}
        variants={fadeIn}
        transition={{ duration: 0.5 }}
        className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-[hsl(var(--primary-blue))] to-[hsl(var(--primary-blue-darker))] px-4 sm:px-6 py-12 sm:py-16 text-white md:px-12 md:py-24"
      >
        <div className="absolute inset-0 bg-[url('/placeholder.svg?height=600&width=1200')] opacity-10 mix-blend-overlay"></div>
        <div className="absolute inset-0">
          <ParticleBackground variant="hero" density={getParticleDensity()} />
          <div className="absolute inset-0 opacity-40">
            <ParticleBackground variant="cosmic" density="low" />
          </div>
          <div className="absolute inset-0 opacity-20">
            <ParticleBackground variant="nebula" density="low" />
          </div>
        </div>
        <div className="relative z-10 grid gap-8 md:grid-cols-2 md:gap-12">
          <motion.div className="space-y-6" variants={fadeIn} transition={{ duration: 0.5, delay: 0.2 }}>
            <motion.h1
              className="text-3xl sm:text-4xl font-bold tracking-tight md:text-5xl lg:text-6xl"
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.7, delay: 0.3 }}
            >
              Find Your Perfect University Path
            </motion.h1>
            <motion.p
              className="max-w-[600px] text-base sm:text-lg text-white/90 md:text-xl"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.7, delay: 0.5 }}
            >
              GradeUP helps you discover the ideal university and program based on your unique skills, interests, and
              goals. Our AI-powered platform provides personalized recommendations to shape your future.
            </motion.p>
            <motion.div
              className="flex flex-wrap gap-3 sm:gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.7 }}
            >
              <Button
                asChild
                size="lg"
                className="bg-white text-[#261FB3] hover:bg-white/90 transition-all duration-300 hover:scale-105 font-medium"
              >
                <Link href="/survey">
                  Take the Survey
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button
                asChild
                size="lg"
                variant="outline"
                className="border-white bg-[hsl(var(--primary-blue))]/20 text-white hover:bg-white/20 transition-all duration-300 hover:scale-105"
              >
                <Link href="/universities">Explore Universities</Link>
              </Button>
            </motion.div>

            <motion.div
              className="flex flex-wrap gap-3 sm:gap-4 mt-6 sm:mt-8"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.9 }}
            >
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 sm:h-5 sm:w-5 text-[hsl(var(--secondary-peach))]" />
                <span className="text-sm sm:text-base text-white/90">Personalized Recommendations</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 sm:h-5 sm:w-5 text-[hsl(var(--secondary-peach))]" />
                <span className="text-sm sm:text-base text-white/90">AI-Powered Matching</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 sm:h-5 sm:w-5 text-[hsl(var(--secondary-peach))]" />
                <span className="text-sm sm:text-base text-white/90">Comprehensive University Data</span>
              </div>
            </motion.div>
          </motion.div>
          <motion.div
            className="hidden md:flex md:items-center md:justify-center"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, delay: 0.5 }}
          >
            <div className="relative h-64 w-64 lg:h-80 lg:w-80">
              <Image
                src="/placeholder.svg?height=400&width=400"
                alt="Students"
                fill
                className="object-cover rounded-2xl"
                priority
              />
              <motion.div
                className="absolute inset-0 rounded-2xl border-4 border-white/30"
                animate={{
                  boxShadow: ["0 0 0 0 rgba(255,255,255,0.3)", "0 0 0 10px rgba(255,255,255,0)"],
                }}
                transition={{
                  repeat: Number.POSITIVE_INFINITY,
                  duration: 2,
                }}
              />
            </div>
          </motion.div>
        </div>

        {/* Floating stats */}
        <motion.div
          className="absolute bottom-4 sm:bottom-6 left-0 right-0 hidden md:flex justify-center gap-4 lg:gap-8 z-20"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 1.1 }}
        >
          <motion.div
            className="glass px-4 py-2 lg:px-6 lg:py-3 rounded-xl flex items-center gap-2 lg:gap-3"
            whileHover={{ y: -5 }}
          >
            <School className="h-4 w-4 lg:h-5 lg:w-5 text-[hsl(var(--secondary-peach))]" />
            <div>
              <span className="text-base lg:text-lg font-bold">10+</span>
              <span className="ml-1 text-xs lg:text-sm">Universities</span>
            </div>
          </motion.div>

          <motion.div
            className="glass px-4 py-2 lg:px-6 lg:py-3 rounded-xl flex items-center gap-2 lg:gap-3"
            whileHover={{ y: -5 }}
          >
            <BookOpen className="h-4 w-4 lg:h-5 lg:w-5 text-[hsl(var(--secondary-peach))]" />
            <div>
              <span className="text-base lg:text-lg font-bold">50+</span>
              <span className="ml-1 text-xs lg:text-sm">Programs</span>
            </div>
          </motion.div>

          <motion.div
            className="glass px-4 py-2 lg:px-6 lg:py-3 rounded-xl flex items-center gap-2 lg:gap-3"
            whileHover={{ y: -5 }}
          >
            <Users className="h-4 w-4 lg:h-5 lg:w-5 text-[hsl(var(--secondary-peach))]" />
            <div>
              <span className="text-base lg:text-lg font-bold">1000+</span>
              <span className="ml-1 text-xs lg:text-sm">Students</span>
            </div>
          </motion.div>
        </motion.div>

        {/* Mobile stats - visible only on small screens */}
        <motion.div
          className="grid grid-cols-3 gap-2 mt-8 md:hidden"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.9 }}
        >
          <div className="glass px-2 py-2 rounded-lg flex flex-col items-center justify-center text-center">
            <School className="h-4 w-4 text-[hsl(var(--secondary-peach))]" />
            <div className="mt-1">
              <div className="text-sm font-bold">10+</div>
              <div className="text-xs">Universities</div>
            </div>
          </div>

          <div className="glass px-2 py-2 rounded-lg flex flex-col items-center justify-center text-center">
            <BookOpen className="h-4 w-4 text-[hsl(var(--secondary-peach))]" />
            <div className="mt-1">
              <div className="text-sm font-bold">50+</div>
              <div className="text-xs">Programs</div>
            </div>
          </div>

          <div className="glass px-2 py-2 rounded-lg flex flex-col items-center justify-center text-center">
            <Users className="h-4 w-4 text-[hsl(var(--secondary-peach))]" />
            <div className="mt-1">
              <div className="text-sm font-bold">1000+</div>
              <div className="text-xs">Students</div>
            </div>
          </div>
        </motion.div>
      </motion.section>

      {/* Features Section */}
      <motion.section
        ref={featuresRef}
        initial="hidden"
        animate={featuresInView ? "visible" : "hidden"}
        variants={staggerContainer}
        className="space-y-8"
      >
        <motion.div className="text-center px-4" variants={fadeIn}>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight md:text-4xl text-[hsl(var(--primary-blue))]">
            How GradeUP Works
          </h2>
          <p className="mt-4 text-muted-foreground max-w-2xl mx-auto">
            Our platform helps you make informed decisions about your academic future through a simple, data-driven
            process that matches your profile with the right educational opportunities.
          </p>
        </motion.div>
        <div className="grid gap-4 sm:gap-6 grid-cols-2 md:grid-cols-4 px-4">
          <motion.div variants={cardVariants} whileHover="hover">
            <Card className="border-2 border-[hsl(var(--secondary-peach))] transition-all h-full">
              <CardContent className="flex flex-col items-center p-3 sm:p-6 text-center">
                <motion.div
                  className="mb-3 sm:mb-4 rounded-full bg-[hsl(var(--secondary-peach))] p-2 sm:p-3"
                  whileHover={{ scale: 1.1, backgroundColor: "hsl(var(--primary-blue))" }}
                  transition={{ duration: 0.3 }}
                >
                  <Users className="h-4 w-4 sm:h-6 sm:w-6 text-[hsl(var(--primary-blue))]" />
                </motion.div>
                <h3 className="mb-1 sm:mb-2 text-base sm:text-xl font-medium">Complete Survey</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  Answer questions about your interests and skills
                </p>
              </CardContent>
            </Card>
          </motion.div>
          <motion.div variants={cardVariants} whileHover="hover">
            <Card className="border-2 border-[hsl(var(--secondary-peach))] transition-all h-full">
              <CardContent className="flex flex-col items-center p-3 sm:p-6 text-center">
                <motion.div
                  className="mb-3 sm:mb-4 rounded-full bg-[hsl(var(--secondary-peach))] p-2 sm:p-3"
                  whileHover={{ scale: 1.1, backgroundColor: "hsl(var(--primary-blue))" }}
                  transition={{ duration: 0.3 }}
                >
                  <School className="h-4 w-4 sm:h-6 sm:w-6 text-[hsl(var(--primary-blue))]" />
                </motion.div>
                <h3 className="mb-1 sm:mb-2 text-base sm:text-xl font-medium">Get Recommendations</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">Receive personalized university suggestions</p>
              </CardContent>
            </Card>
          </motion.div>
          <motion.div variants={cardVariants} whileHover="hover">
            <Card className="border-2 border-[hsl(var(--secondary-peach))] transition-all h-full">
              <CardContent className="flex flex-col items-center p-3 sm:p-6 text-center">
                <motion.div
                  className="mb-3 sm:mb-4 rounded-full bg-[hsl(var(--secondary-peach))] p-2 sm:p-3"
                  whileHover={{ scale: 1.1, backgroundColor: "hsl(var(--primary-blue))" }}
                  transition={{ duration: 0.3 }}
                >
                  <BookOpen className="h-4 w-4 sm:h-6 sm:w-6 text-[hsl(var(--primary-blue))]" />
                </motion.div>
                <h3 className="mb-1 sm:mb-2 text-base sm:text-xl font-medium">Explore Options</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">Browse detailed university information</p>
              </CardContent>
            </Card>
          </motion.div>
          <motion.div variants={cardVariants} whileHover="hover">
            <Card className="border-2 border-[hsl(var(--secondary-peach))] transition-all h-full">
              <CardContent className="flex flex-col items-center p-3 sm:p-6 text-center">
                <motion.div
                  className="mb-3 sm:mb-4 rounded-full bg-[hsl(var(--secondary-peach))] p-2 sm:p-3"
                  whileHover={{ scale: 1.1, backgroundColor: "hsl(var(--primary-blue))" }}
                  transition={{ duration: 0.3 }}
                >
                  <GraduationCap className="h-4 w-4 sm:h-6 sm:w-6 text-[hsl(var(--primary-blue))]" />
                </motion.div>
                <h3 className="mb-1 sm:mb-2 text-base sm:text-xl font-medium">Make Decisions</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">Choose the best path for your goals</p>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </motion.section>

      {/* Detailed Process Timeline */}
      <motion.section
        ref={timelineRef}
        initial="hidden"
        animate={timelineInView ? "visible" : "hidden"}
        variants={fadeIn}
        className="py-10 px-4"
      >
        <motion.div className="text-center mb-12" variants={fadeIn} transition={{ duration: 0.5 }}>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight md:text-4xl text-[hsl(var(--primary-blue))]">
            Your Journey to the Perfect University
          </h2>
          <p className="mt-4 text-muted-foreground max-w-2xl mx-auto">
            Follow these steps to discover and secure your ideal educational path
          </p>
        </motion.div>

        <div className="relative">
          {/* Timeline line */}
          <motion.div
            className="absolute left-4 top-0 bottom-0 w-0.5 bg-[hsl(var(--primary-blue))] md:left-1/2 md:-ml-0.5"
            initial={{ height: 0 }}
            animate={timelineInView ? { height: "100%" } : { height: 0 }}
            transition={{ duration: 1.5 }}
          ></motion.div>

          {/* Timeline items */}
          <div className="space-y-8 sm:space-y-12">
            {timeline.map((item, index) => (
              <motion.div
                key={index}
                className="relative"
                initial={{ opacity: 0, y: 50 }}
                animate={timelineInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
              >
                <div className={`flex flex-col md:flex-row ${index % 2 === 0 ? "md:flex-row-reverse" : ""}`}>
                  <div className="md:w-1/2"></div>
                  <motion.div
                    className="z-10 flex h-6 w-6 sm:h-8 sm:w-8 items-center justify-center rounded-full bg-[hsl(var(--primary-blue))] text-white shadow-md absolute left-1 md:left-1/2 md:-ml-3 sm:md:-ml-4"
                    initial={{ scale: 0 }}
                    animate={timelineInView ? { scale: 1 } : { scale: 0 }}
                    transition={{ duration: 0.3, delay: 0.2 + index * 0.2 }}
                    whileHover={{ scale: 1.2 }}
                  >
                    <item.icon className="h-3 w-3 sm:h-4 sm:w-4" />
                  </motion.div>
                  <div className="md:w-1/2">
                    <div className={`ml-10 md:ml-0 ${index % 2 === 0 ? "md:mr-10 lg:mr-12" : "md:ml-10 lg:ml-12"}`}>
                      <div className="rounded-lg bg-card p-4 sm:p-6 shadow-md">
                        <h3 className="text-base sm:text-xl font-bold text-[hsl(var(--primary-blue))]">{item.title}</h3>
                        <p className="mt-1 sm:mt-2 text-xs sm:text-sm text-muted-foreground">{item.description}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      {/* Stats Section with Animated Counters */}
      <motion.section
        ref={statsRef}
        initial="hidden"
        animate={statsInView ? "visible" : "hidden"}
        variants={fadeIn}
        transition={{ duration: 0.5 }}
        className="rounded-xl animated-gradient p-6 sm:p-8 text-white md:p-12 relative overflow-hidden"
      >
        <div className="absolute inset-0 opacity-20">
          <ParticleBackground variant="dense" density={getParticleDensity()} />
        </div>
        <div className="grid gap-6 sm:gap-8 grid-cols-3 relative z-10">
          <motion.div
            className="text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={statsInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <motion.div
              className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold"
              initial={{ scale: 0.5 }}
              animate={statsInView ? { scale: 1 } : { scale: 0.5 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <AnimatedCounter value={10} duration={1.5} />+
            </motion.div>
            <p className="mt-1 sm:mt-2 text-xs sm:text-sm text-white/80">Universities</p>
            <motion.div
              className="mt-2 sm:mt-4 mx-auto w-10 sm:w-16 h-1 bg-[hsl(var(--secondary-peach))] rounded-full"
              initial={{ width: 0 }}
              animate={statsInView ? { width: windowWidth < 640 ? "2.5rem" : "4rem" } : { width: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            />
          </motion.div>
          <motion.div
            className="text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={statsInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <motion.div
              className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold"
              initial={{ scale: 0.5 }}
              animate={statsInView ? { scale: 1 } : { scale: 0.5 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <AnimatedCounter value={50} duration={1.5} />+
            </motion.div>
            <p className="mt-1 sm:mt-2 text-xs sm:text-sm text-white/80">Programs</p>
            <motion.div
              className="mt-2 sm:mt-4 mx-auto w-10 sm:w-16 h-1 bg-[hsl(var(--secondary-peach))] rounded-full"
              initial={{ width: 0 }}
              animate={statsInView ? { width: windowWidth < 640 ? "2.5rem" : "4rem" } : { width: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
            />
          </motion.div>
          <motion.div
            className="text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={statsInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            <motion.div
              className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold"
              initial={{ scale: 0.5 }}
              animate={statsInView ? { scale: 1 } : { scale: 0.5 }}
              transition={{ duration: 0.5, delay: 0.5 }}
            >
              <AnimatedCounter value={1000} duration={1.5} />+
            </motion.div>
            <p className="mt-1 sm:mt-2 text-xs sm:text-sm text-white/80">Students Helped</p>
            <motion.div
              className="mt-2 sm:mt-4 mx-auto w-10 sm:w-16 h-1 bg-[hsl(var(--secondary-peach))] rounded-full"
              initial={{ width: 0 }}
              animate={statsInView ? { width: windowWidth < 640 ? "2.5rem" : "4rem" } : { width: 0 }}
              transition={{ duration: 0.5, delay: 0.7 }}
            />
          </motion.div>
        </div>

        <motion.div
          className="mt-8 sm:mt-12 grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-6 relative z-10"
          initial={{ opacity: 0 }}
          animate={statsInView ? { opacity: 1 } : { opacity: 0 }}
          transition={{ duration: 0.5, delay: 0.7 }}
        >
          <div className="glass rounded-lg p-3 sm:p-4 text-center">
            <div className="flex justify-center mb-2 sm:mb-3">
              <Award className="h-4 w-4 sm:h-6 sm:w-6 text-[hsl(var(--secondary-peach))]" />
            </div>
            <h4 className="text-sm sm:text-base font-medium">Top-Rated</h4>
            <p className="text-xs sm:text-sm text-white/80">Universities</p>
          </div>

          <div className="glass rounded-lg p-3 sm:p-4 text-center">
            <div className="flex justify-center mb-2 sm:mb-3">
              <Star className="h-4 w-4 sm:h-6 sm:w-6 text-[hsl(var(--secondary-peach))]" />
            </div>
            <h4 className="text-sm sm:text-base font-medium">95% Satisfaction</h4>
            <p className="text-xs sm:text-sm text-white/80">From Students</p>
          </div>

          <div className="glass rounded-lg p-3 sm:p-4 text-center">
            <div className="flex justify-center mb-2 sm:mb-3">
              <TrendingUp className="h-4 w-4 sm:h-6 sm:w-6 text-[hsl(var(--secondary-peach))]" />
            </div>
            <h4 className="text-sm sm:text-base font-medium">87% Success</h4>
            <p className="text-xs sm:text-sm text-white/80">Admission Rate</p>
          </div>

          <div className="glass rounded-lg p-3 sm:p-4 text-center">
            <div className="flex justify-center mb-2 sm:mb-3">
              <Clock className="h-4 w-4 sm:h-6 sm:w-6 text-[hsl(var(--secondary-peach))]" />
            </div>
            <h4 className="text-sm sm:text-base font-medium">15 Minutes</h4>
            <p className="text-xs sm:text-sm text-white/80">Average Survey Time</p>
          </div>
        </motion.div>
      </motion.section>

      {/* Benefits & Features Tabs */}
      <section className="py-10 px-4">
        <div className="text-center mb-8 sm:mb-12">
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight md:text-4xl text-[hsl(var(--primary-blue))]">
            Why Choose GradeUP
          </h2>
          <p className="mt-4 text-muted-foreground max-w-2xl mx-auto">
            Discover the benefits and features that make GradeUP the leading platform for educational guidance
          </p>
        </div>

        <Tabs defaultValue="students" className="w-full">
          <TabsList className="grid w-full grid-cols-3 mb-6 sm:mb-8">
            <TabsTrigger value="students">For Students</TabsTrigger>
            <TabsTrigger value="parents">For Parents</TabsTrigger>
            <TabsTrigger value="educators">For Educators</TabsTrigger>
          </TabsList>

          <TabsContent value="students" className="space-y-4">
            <div className="grid md:grid-cols-2 gap-6 sm:gap-8">
              <motion.div
                className="relative overflow-hidden rounded-xl h-48 sm:h-auto"
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Image
                  src="/placeholder.svg?height=400&width=600"
                  alt="Students using GradeUP"
                  width={600}
                  height={400}
                  className="rounded-xl object-cover w-full h-full"
                />
              </motion.div>

              <motion.div
                className="space-y-4 sm:space-y-6"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <h3 className="text-xl sm:text-2xl font-bold text-[hsl(var(--primary-blue))]">
                  Empowering Student Choices
                </h3>
                <p className="text-sm sm:text-base text-muted-foreground">
                  GradeUP helps students navigate the complex world of higher education by providing personalized
                  guidance based on their unique profile, interests, and goals.
                </p>

                <div className="space-y-3 sm:space-y-4">
                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">Personalized Recommendations</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Get university and program suggestions tailored to your specific profile
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">Comprehensive University Data</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Access detailed information about programs, facilities, and admission requirements
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">Skills Assessment</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Discover your strengths and how they align with different academic paths
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">Career Path Insights</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Understand how your educational choices connect to future career opportunities
                      </p>
                    </div>
                  </div>
                </div>

                <Button asChild className="bg-[hsl(var(--primary-blue))] hover:bg-[hsl(var(--primary-blue-dark))] mt-4">
                  <Link href="/survey">
                    Start Your Journey
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </motion.div>
            </div>
          </TabsContent>

          <TabsContent value="parents" className="space-y-4">
            <div className="grid md:grid-cols-2 gap-6 sm:gap-8">
              <motion.div
                className="relative overflow-hidden rounded-xl h-48 sm:h-auto"
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Image
                  src="/placeholder.svg?height=400&width=600"
                  alt="Parents using GradeUP"
                  width={600}
                  height={400}
                  className="rounded-xl object-cover w-full h-full"
                />
              </motion.div>

              <motion.div
                className="space-y-4 sm:space-y-6"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <h3 className="text-xl sm:text-2xl font-bold text-[hsl(var(--primary-blue))]">
                  Supporting Informed Decisions
                </h3>
                <p className="text-sm sm:text-base text-muted-foreground">
                  GradeUP provides parents with the tools and information needed to guide their children through the
                  important decision of choosing the right educational path.
                </p>

                <div className="space-y-3 sm:space-y-4">
                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">Data-Driven Guidance</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Access objective information to help your child make the best educational choices
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">Financial Planning Tools</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Understand tuition costs, scholarship opportunities, and financial aid options
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">University Comparison</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Compare different institutions side by side to evaluate the best options
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">Future Prospects</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Understand the career opportunities and outcomes associated with different programs
                      </p>
                    </div>
                  </div>
                </div>

                <Button asChild className="bg-[hsl(var(--primary-blue))] hover:bg-[hsl(var(--primary-blue-dark))] mt-4">
                  <Link href="/survey">
                    Support Your Child
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </motion.div>
            </div>
          </TabsContent>

          <TabsContent value="educators" className="space-y-4">
            <div className="grid md:grid-cols-2 gap-6 sm:gap-8">
              <motion.div
                className="relative overflow-hidden rounded-xl h-48 sm:h-auto"
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Image
                  src="/placeholder.svg?height=400&width=600"
                  alt="Educators using GradeUP"
                  width={600}
                  height={400}
                  className="rounded-xl object-cover w-full h-full"
                />
              </motion.div>

              <motion.div
                className="space-y-4 sm:space-y-6"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <h3 className="text-xl sm:text-2xl font-bold text-[hsl(var(--primary-blue))]">
                  Enhancing Educational Guidance
                </h3>
                <p className="text-sm sm:text-base text-muted-foreground">
                  GradeUP provides educators and counselors with powerful tools to better guide students through their
                  educational journey and career planning.
                </p>

                <div className="space-y-3 sm:space-y-4">
                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">Student Profile Insights</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Gain deeper understanding of student strengths, interests, and potential paths
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">Comprehensive Resources</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Access up-to-date information about universities, programs, and admission requirements
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">Group Management</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Track and manage multiple students' educational planning processes
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <div className="rounded-full bg-[hsl(var(--secondary-peach))] p-1.5 sm:p-2 h-fit">
                      <CheckCircle className="h-3.5 w-3.5 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
                    </div>
                    <div>
                      <h4 className="text-sm sm:text-base font-medium">Data-Driven Counseling</h4>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Use objective data to provide better guidance and recommendations
                      </p>
                    </div>
                  </div>
                </div>

                <Button asChild className="bg-[hsl(var(--primary-blue))] hover:bg-[hsl(var(--primary-blue-dark))] mt-4">
                  <Link href="/register">
                    Join as Educator
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </motion.div>
            </div>
          </TabsContent>
        </Tabs>
      </section>

      {/* Testimonials Section */}
      <motion.section
        ref={testimonialsRef}
        initial="hidden"
        animate={testimonialsInView ? "visible" : "hidden"}
        variants={fadeIn}
        transition={{ duration: 0.5 }}
        className="py-10 px-4"
      >
        <motion.div className="text-center mb-8 sm:mb-12" variants={fadeIn} transition={{ duration: 0.5 }}>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight md:text-4xl text-[hsl(var(--primary-blue))]">
            What Our Users Say
          </h2>
          <p className="mt-4 text-muted-foreground max-w-2xl mx-auto">
            Hear from students who found their perfect educational path with GradeUP
          </p>
        </motion.div>

        <div className="relative overflow-hidden">
          <div className="mx-auto max-w-3xl px-4">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTestimonial}
                initial={{ opacity: 0, x: 100 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -100 }}
                transition={{ duration: 0.5 }}
                className="relative rounded-xl bg-card p-6 sm:p-8 shadow-lg border border-[hsl(var(--secondary-peach))]"
              >
                <div className="mb-4 sm:mb-6 flex items-center gap-3 sm:gap-4">
                  <div className="relative h-12 w-12 sm:h-16 sm:w-16 overflow-hidden rounded-full">
                    <Image
                      src={testimonials[activeTestimonial].avatar || "/placeholder.svg"}
                      alt={testimonials[activeTestimonial].name}
                      fill
                      className="object-cover"
                    />
                  </div>
                  <div>
                    <h3 className="text-base sm:text-lg font-bold">{testimonials[activeTestimonial].name}</h3>
                    <p className="text-xs sm:text-sm text-muted-foreground">{testimonials[activeTestimonial].role}</p>
                    <p className="text-xs sm:text-sm text-[hsl(var(--primary-blue))]">
                      {testimonials[activeTestimonial].university}
                    </p>
                  </div>
                </div>
                <div className="relative">
                  <svg
                    className="absolute -left-2 -top-2 sm:-left-3 sm:-top-3 h-6 w-6 sm:h-8 sm:w-8 text-[hsl(var(--secondary-peach))] opacity-50"
                    fill="currentColor"
                    viewBox="0 0 32 32"
                    aria-hidden="true"
                  >
                    <path d="M9.352 4C4.456 7.456 1 13.12 1 19.36c0 5.088 3.072 8.064 6.624 8.064 3.36 0 5.856-2.688 5.856-5.856 0-3.168-2.208-5.472-5.088-5.472-.576 0-1.344.096-1.536.192.48-3.264 3.552-7.104 6.624-9.024L9.352 4zm16.512 0c-4.8 3.456-8.256 9.12-8.256 15.36 0 5.088 3.072 8.064 6.624 8.064 3.264 0 5.856-2.688 5.856-5.856 0-3.168-2.304-5.472-5.184-5.472-.576 0-1.248.096-1.44.192.48-3.264 3.456-7.104 6.528-9.024L25.864 4z" />
                  </svg>
                  <p className="relative text-sm sm:text-lg italic text-muted-foreground">
                    "{testimonials[activeTestimonial].quote}"
                  </p>
                </div>

                {/* Navigation buttons */}
                <div className="absolute top-1/2 -left-4 -translate-y-1/2 hidden sm:block">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="rounded-full bg-background/80 backdrop-blur-sm hover:bg-background"
                    onClick={prevTestimonial}
                    aria-label="Previous testimonial"
                  >
                    <ChevronLeft className="h-5 w-5" />
                  </Button>
                </div>
                <div className="absolute top-1/2 -right-4 -translate-y-1/2 hidden sm:block">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="rounded-full bg-background/80 backdrop-blur-sm hover:bg-background"
                    onClick={nextTestimonial}
                    aria-label="Next testimonial"
                  >
                    <ChevronRight className="h-5 w-5" />
                  </Button>
                </div>
              </motion.div>
            </AnimatePresence>

            <div className="mt-6 sm:mt-8 flex justify-center gap-2">
              {testimonials.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setActiveTestimonial(index)}
                  className={`h-2 sm:h-2.5 w-2 sm:w-2.5 rounded-full transition-colors ${
                    index === activeTestimonial ? "bg-[hsl(var(--primary-blue))]" : "bg-[hsl(var(--secondary-peach))]"
                  }`}
                  aria-label={`Go to testimonial ${index + 1}`}
                />
              ))}
            </div>

            {/* Mobile navigation buttons */}
            <div className="mt-4 flex justify-center gap-4 sm:hidden">
              <Button
                variant="outline"
                size="sm"
                className="h-8 w-8 rounded-full p-0"
                onClick={prevTestimonial}
                aria-label="Previous testimonial"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-8 w-8 rounded-full p-0"
                onClick={nextTestimonial}
                aria-label="Next testimonial"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </motion.section>

      {/* FAQ Section */}
      <motion.section
        ref={faqRef}
        initial="hidden"
        animate={faqInView ? "visible" : "hidden"}
        variants={fadeIn}
        transition={{ duration: 0.5 }}
        className="py-10 px-4"
      >
        <motion.div className="text-center mb-8 sm:mb-12" variants={fadeIn} transition={{ duration: 0.5 }}>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight md:text-4xl text-[hsl(var(--primary-blue))]">
            Frequently Asked Questions
          </h2>
          <p className="mt-4 text-muted-foreground max-w-2xl mx-auto">
            Find answers to common questions about GradeUP and our services
          </p>
        </motion.div>

        <div className="mx-auto max-w-3xl">
          <Accordion type="single" collapsible className="w-full">
            {faqs.map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={faqInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <AccordionItem value={`item-${index}`}>
                  <AccordionTrigger className="text-left text-sm sm:text-base hover:text-[hsl(var(--primary-blue))]">
                    {faq.question}
                  </AccordionTrigger>
                  <AccordionContent className="text-xs sm:text-sm text-muted-foreground">{faq.answer}</AccordionContent>
                </AccordionItem>
              </motion.div>
            ))}
          </Accordion>
        </div>
      </motion.section>

      {/* CTA Section */}
      <motion.section
        ref={ctaRef}
        initial="hidden"
        animate={ctaInView ? "visible" : "hidden"}
        variants={fadeIn}
        transition={{ duration: 0.5 }}
        className="rounded-xl bg-gradient-to-r from-[hsl(var(--secondary-peach))] to-white p-6 sm:p-8 md:p-12"
      >
        <div className="grid gap-6 sm:gap-8 md:grid-cols-2 md:gap-12">
          <motion.div className="space-y-4" variants={fadeIn} transition={{ duration: 0.5, delay: 0.2 }}>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-[hsl(var(--primary-blue))] md:text-4xl">
              Ready to Find Your Path?
            </h2>
            <p className="text-sm sm:text-base text-muted-foreground">
              Take our comprehensive survey and get personalized university recommendations tailored to your unique
              profile, interests, and goals.
            </p>
            <div className="flex flex-wrap gap-3 sm:gap-4">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button
                  asChild
                  size="lg"
                  className="bg-[hsl(var(--primary-blue))] hover:bg-[hsl(var(--primary-blue-dark))] transition-all duration-300"
                >
                  <Link href="/survey">
                    Start Now
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </motion.div>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button
                  asChild
                  size="lg"
                  variant="outline"
                  className="border-[hsl(var(--primary-blue))] text-[hsl(var(--primary-blue))] hover:bg-[hsl(var(--primary-blue))]/10"
                >
                  <Link href="/universities">Explore Universities</Link>
                </Button>
              </motion.div>
            </div>

            <div className="mt-4 sm:mt-6 flex items-center gap-2">
              <ThumbsUp className="h-4 w-4 sm:h-5 sm:w-5 text-[hsl(var(--primary-blue))]" />
              <span className="text-xs sm:text-sm font-medium">
                Join over 1,000 students who found their perfect university match
              </span>
            </div>
          </motion.div>
          <motion.div
            className="hidden md:block"
            initial={{ opacity: 0, x: 50 }}
            animate={ctaInView ? { opacity: 1, x: 0 } : { opacity: 0, x: 50 }}
            transition={{ duration: 0.7, delay: 0.3 }}
          >
            <motion.div whileHover={{ scale: 1.03, rotate: 1 }} transition={{ duration: 0.3 }} className="relative">
              <Image
                src="/placeholder.svg?height=300&width=500"
                alt="Students in university"
                width={500}
                height={300}
                className="rounded-xl object-cover shadow-lg"
              />
              <motion.div
                className="absolute -bottom-4 -right-4 rounded-lg bg-[hsl(var(--primary-blue))] p-4 text-white shadow-lg"
                initial={{ scale: 0 }}
                animate={ctaInView ? { scale: 1 } : { scale: 0 }}
                transition={{ duration: 0.5, delay: 0.7 }}
                whileHover={{ scale: 1.1 }}
              >
                <div className="flex items-center gap-2">
                  <Heart className="h-5 w-5 text-[hsl(var(--secondary-peach))]" />
                  <span className="font-medium">95% Satisfaction Rate</span>
                </div>
              </motion.div>
            </motion.div>
          </motion.div>
        </div>
      </motion.section>
    </div>
  )
}
