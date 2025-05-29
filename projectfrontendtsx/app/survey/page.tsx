"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ArrowLeft, ArrowRight, CheckCircle, Telescope, Sparkles } from "lucide-react"
import { useRouter } from "next/navigation"
import { ParticleBackground } from "@/components/particle-background"
import { motion } from "framer-motion"

// Survey questions
const surveyQuestions = [
  {
    id: "academic-interests",
    title: "Academic Interests",
    description: "Tell us about your academic preferences",
    questions: [
      {
        id: "favorite-subjects",
        type: "checkbox",
        question: "Which subjects do you enjoy the most?",
        options: [
          { id: "math", label: "Mathematics" },
          { id: "science", label: "Science" },
          { id: "literature", label: "Literature" },
          { id: "history", label: "History" },
          { id: "arts", label: "Arts" },
          { id: "technology", label: "Technology" },
          { id: "languages", label: "Languages" },
          { id: "physical-education", label: "Physical Education" },
        ],
      },
      {
        id: "learning-style",
        type: "radio",
        question: "How do you prefer to learn?",
        options: [
          { id: "visual", label: "Visual (images, diagrams)" },
          { id: "auditory", label: "Auditory (listening, discussing)" },
          { id: "reading", label: "Reading/Writing" },
          { id: "kinesthetic", label: "Kinesthetic (hands-on activities)" },
        ],
      },
      {
        id: "study-environment",
        type: "radio",
        question: "What study environment do you prefer?",
        options: [
          { id: "quiet", label: "Quiet, individual study" },
          { id: "group", label: "Group study and collaboration" },
          { id: "mixed", label: "Mix of both, depending on the task" },
        ],
      },
    ],
  },
  {
    id: "skills-abilities",
    title: "Skills & Abilities",
    description: "Assess your skills and abilities",
    questions: [
      {
        id: "analytical-thinking",
        type: "slider",
        question: "How would you rate your analytical thinking skills?",
        min: 1,
        max: 10,
        step: 1,
      },
      {
        id: "creativity",
        type: "slider",
        question: "How would you rate your creativity?",
        min: 1,
        max: 10,
        step: 1,
      },
      {
        id: "communication",
        type: "slider",
        question: "How would you rate your communication skills?",
        min: 1,
        max: 10,
        step: 1,
      },
      {
        id: "teamwork",
        type: "slider",
        question: "How would you rate your teamwork abilities?",
        min: 1,
        max: 10,
        step: 1,
      },
    ],
  },
  {
    id: "career-goals",
    title: "Career Goals",
    description: "Tell us about your career aspirations",
    questions: [
      {
        id: "career-interest",
        type: "checkbox",
        question: "Which career fields interest you the most?",
        options: [
          { id: "technology", label: "Technology & IT" },
          { id: "healthcare", label: "Healthcare & Medicine" },
          { id: "business", label: "Business & Finance" },
          { id: "arts", label: "Arts & Design" },
          { id: "education", label: "Education & Teaching" },
          { id: "engineering", label: "Engineering" },
          { id: "science", label: "Science & Research" },
          { id: "social-services", label: "Social Services" },
        ],
      },
      {
        id: "work-environment",
        type: "radio",
        question: "What type of work environment do you prefer?",
        options: [
          { id: "office", label: "Traditional office setting" },
          { id: "remote", label: "Remote/work from home" },
          { id: "field", label: "Field work/outdoors" },
          { id: "mixed", label: "Mixed environment" },
        ],
      },
      {
        id: "work-values",
        type: "checkbox",
        question: "What do you value most in a career?",
        options: [
          { id: "salary", label: "High salary" },
          { id: "work-life", label: "Work-life balance" },
          { id: "growth", label: "Growth opportunities" },
          { id: "impact", label: "Making a positive impact" },
          { id: "stability", label: "Job stability" },
          { id: "creativity", label: "Creative expression" },
          { id: "autonomy", label: "Autonomy and independence" },
        ],
      },
    ],
  },
]

export default function SurveyPage() {
  const router = useRouter()
  const [currentSection, setCurrentSection] = useState(0)
  const [answers, setAnswers] = useState({})
  const [progress, setProgress] = useState(0)

  const handleCheckboxChange = (questionId, optionId, checked) => {
    setAnswers((prev) => {
      const currentAnswers = prev[questionId] || []
      if (checked) {
        return {
          ...prev,
          [questionId]: [...currentAnswers, optionId],
        }
      } else {
        return {
          ...prev,
          [questionId]: currentAnswers.filter((id) => id !== optionId),
        }
      }
    })
    updateProgress()
  }

  const handleRadioChange = (questionId, value) => {
    setAnswers((prev) => ({
      ...prev,
      [questionId]: value,
    }))
    updateProgress()
  }

  const handleSliderChange = (questionId, value) => {
    setAnswers((prev) => ({
      ...prev,
      [questionId]: value[0],
    }))
    updateProgress()
  }

  const updateProgress = () => {
    const totalQuestions = surveyQuestions.reduce((acc, section) => acc + section.questions.length, 0)

    let answeredQuestions = 0
    surveyQuestions.forEach((section) => {
      section.questions.forEach((question) => {
        if (answers[question.id]) {
          if (Array.isArray(answers[question.id])) {
            if (answers[question.id].length > 0) answeredQuestions++
          } else {
            answeredQuestions++
          }
        }
      })
    })

    setProgress(Math.round((answeredQuestions / totalQuestions) * 100))
  }

  const nextSection = () => {
    if (currentSection < surveyQuestions.length - 1) {
      setCurrentSection(currentSection + 1)
      window.scrollTo(0, 0)
    } else {
      router.push("/survey/results")
    }
  }

  const prevSection = () => {
    if (currentSection > 0) {
      setCurrentSection(currentSection - 1)
      window.scrollTo(0, 0)
    }
  }

  const currentSectionData = surveyQuestions[currentSection]

  return (
    <div className="flex flex-col items-center w-full min-h-[calc(100vh-4rem)] py-10 px-4">
      {/* Multiple particle layers for cosmic depth */}
      <div className="particle-container fixed inset-0 z-0">
        <ParticleBackground variant="starfield" density="medium" />
      </div>
      <div className="particle-container fixed inset-0 z-0 opacity-50">
        <ParticleBackground variant="cosmic" density="low" />
      </div>
      <div className="particle-container fixed inset-0 z-0 opacity-30">
        <ParticleBackground variant="nebula" density="low" />
      </div>
      <div className="particle-container fixed inset-0 z-0 opacity-20">
        <ParticleBackground variant="hero" density="low" />
      </div>

      <div className="container mx-auto relative z-10 max-w-4xl">
        <motion.div
          className="mb-8 text-center"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center justify-center mb-4">
            <div className="relative">
              <Telescope className="h-10 w-10 text-cosmic-purple animate-pulse-glow" />
              <Sparkles className="absolute -top-1 -right-1 h-4 w-4 text-cosmic-cyan animate-starfield-twinkle" />
            </div>
          </div>
          <h1 className="text-3xl md:text-4xl font-bold cosmic-text mb-2">Cosmic Educational Survey</h1>
          <p className="text-space-white/70 max-w-2xl mx-auto">
            Navigate through the cosmos of education to discover your perfect university match
          </p>
        </motion.div>

        <motion.div
          className="mb-8"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-space-white/90">Journey Progress</span>
            <span className="text-sm font-medium text-cosmic-cyan">{progress}%</span>
          </div>
          <div className="cosmic-progress h-3">
            <motion.div
              className="cosmic-progress-fill h-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </motion.div>

        <Tabs defaultValue={currentSectionData.id} className="w-full">
          <TabsList className="grid w-full grid-cols-3 mb-8 bg-black/30 border border-white/20">
            {surveyQuestions.map((section, index) => (
              <TabsTrigger
                key={section.id}
                value={section.id}
                onClick={() => setCurrentSection(index)}
                disabled={index > currentSection}
                className={`text-space-white data-[state=active]:bg-cosmic-purple data-[state=active]:text-space-white transition-all duration-300 ${
                  index > currentSection ? "opacity-50" : ""
                }`}
              >
                {section.title}
              </TabsTrigger>
            ))}
          </TabsList>

          <motion.div
            key={currentSectionData.id}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.4 }}
          >
            <Card className="cosmic-card border-2 border-white/20 shadow-2xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-3 cosmic-text text-xl">
                  <div className="p-2 rounded-full bg-cosmic-purple/20 border border-cosmic-purple/30">
                    <Telescope className="h-5 w-5 text-cosmic-purple" />
                  </div>
                  {currentSectionData.title}
                </CardTitle>
                <CardDescription className="text-space-white/70">{currentSectionData.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-8">
                {currentSectionData.questions.map((question, questionIndex) => (
                  <motion.div
                    key={question.id}
                    className="space-y-4"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: questionIndex * 0.1 }}
                  >
                    <h3 className="text-lg font-medium text-space-white">{question.question}</h3>

                    {question.type === "checkbox" && (
                      <div className="grid gap-3 sm:grid-cols-2">
                        {question.options.map((option) => (
                          <div key={option.id} className="flex items-start space-x-3 group">
                            <Checkbox
                              id={`${question.id}-${option.id}`}
                              checked={(answers[question.id] || []).includes(option.id)}
                              onCheckedChange={(checked) => handleCheckboxChange(question.id, option.id, checked)}
                              className="data-[state=checked]:bg-cosmic-purple data-[state=checked]:border-cosmic-purple border-white/30 group-hover:border-cosmic-cyan transition-colors duration-200"
                            />
                            <Label
                              htmlFor={`${question.id}-${option.id}`}
                              className="text-sm font-normal leading-none text-space-white/90 group-hover:text-cosmic-cyan transition-colors duration-200 cursor-pointer"
                            >
                              {option.label}
                            </Label>
                          </div>
                        ))}
                      </div>
                    )}

                    {question.type === "radio" && (
                      <RadioGroup
                        value={answers[question.id] || ""}
                        onValueChange={(value) => handleRadioChange(question.id, value)}
                        className="space-y-3"
                      >
                        {question.options.map((option) => (
                          <div key={option.id} className="flex items-center space-x-3 group">
                            <RadioGroupItem
                              value={option.id}
                              id={`${question.id}-${option.id}`}
                              className="border-white/30 text-cosmic-purple group-hover:border-cosmic-cyan transition-colors duration-200"
                            />
                            <Label
                              htmlFor={`${question.id}-${option.id}`}
                              className="text-sm font-normal text-space-white/90 group-hover:text-cosmic-cyan transition-colors duration-200 cursor-pointer"
                            >
                              {option.label}
                            </Label>
                          </div>
                        ))}
                      </RadioGroup>
                    )}

                    {question.type === "slider" && (
                      <div className="space-y-4">
                        <div className="flex justify-between text-sm text-space-white/70">
                          <span>Low</span>
                          <span>High</span>
                        </div>
                        <Slider
                          defaultValue={[answers[question.id] || 5]}
                          max={question.max}
                          min={question.min}
                          step={question.step}
                          onValueChange={(value) => handleSliderChange(question.id, value)}
                          className="py-4"
                        />
                        <div className="text-center">
                          <span className="inline-flex items-center px-3 py-1 rounded-full bg-cosmic-purple/20 border border-cosmic-purple/30 text-cosmic-purple font-medium">
                            {answers[question.id] || 5}/{question.max}
                          </span>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))}
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button
                  variant="outline"
                  onClick={prevSection}
                  disabled={currentSection === 0}
                  className="gap-2 bg-black/30 border-white/30 text-space-white hover:bg-cosmic-purple/20 hover:border-cosmic-purple transition-all duration-300"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Previous
                </Button>
                <Button onClick={nextSection} className="cosmic-button gap-2 text-space-white font-medium">
                  {currentSection === surveyQuestions.length - 1 ? (
                    <>
                      Launch Results
                      <CheckCircle className="h-4 w-4" />
                    </>
                  ) : (
                    <>
                      Next
                      <ArrowRight className="h-4 w-4" />
                    </>
                  )}
                </Button>
              </CardFooter>
            </Card>
          </motion.div>
        </Tabs>
      </div>
    </div>
  )
}
