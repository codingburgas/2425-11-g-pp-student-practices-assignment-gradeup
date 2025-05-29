"use client"

import { Button } from "@/components/ui/button"
import { ModeToggle } from "@/components/mode-toggle"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { GraduationCap, Menu } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"

export function Header() {
  const pathname = usePathname()
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  // Check if current page is login, register, or survey
  const isAuthOrSurveyPage = ["/login", "/register", "/survey"].some(
    (path) => pathname === path || pathname.startsWith(path + "/"),
  )

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10)
    }

    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen)
  }

  const navLinks = [
    { href: "/", label: "Home" },
    { href: "/survey", label: "Survey" },
    { href: "/universities", label: "Universities" },
    { href: "/specialties", label: "Specialties" },
  ]

  return (
    <header
      className={`sticky top-0 z-50 w-full transition-all duration-300 ${
        isScrolled
          ? "bg-background/80 backdrop-blur-md shadow-md border-b border-cosmic-purple/20"
          : "bg-background/50 backdrop-blur-sm"
      } ${isAuthOrSurveyPage ? "border-b border-cosmic-purple/20" : ""}`}
    >
      <div className="w-full flex h-16 items-center justify-between px-4 md:px-6 max-w-none">
        <div className="flex items-center gap-2">
          {!isAuthOrSurveyPage && <SidebarTrigger className="md:hidden text-cosmic-purple hover:text-cosmic-cyan" />}
          <Link href="/" className="flex items-center gap-2 group">
            <motion.div whileHover={{ rotate: 360 }} transition={{ duration: 0.5 }}>
              <GraduationCap className="h-6 w-6 text-cosmic-purple transition-transform duration-300 group-hover:scale-110 group-hover:text-cosmic-cyan" />
            </motion.div>
            <motion.span
              className="text-xl font-bold text-cosmic-purple group-hover:text-cosmic-cyan transition-colors duration-300"
              initial={{ opacity: 1 }}
              whileHover={{ scale: 1.05 }}
              transition={{ duration: 0.2 }}
            >
              GradeUP
            </motion.span>
          </Link>
        </div>

        <nav className="hidden md:flex md:items-center md:gap-6">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`relative text-sm font-medium transition-colors hover:text-cosmic-purple ${
                pathname === link.href || (link.href !== "/" && pathname.startsWith(link.href))
                  ? "text-cosmic-purple"
                  : "text-foreground hover:text-cosmic-cyan"
              }`}
            >
              {link.label}
              {(pathname === link.href || (link.href !== "/" && pathname.startsWith(link.href))) && (
                <motion.span
                  className="absolute -bottom-1 left-0 h-0.5 w-full bg-cosmic-purple"
                  layoutId="underline"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.3 }}
                />
              )}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-4">
          <ModeToggle />
          <div className="hidden md:flex md:gap-2">
            <Button
              asChild
              variant="outline"
              size="sm"
              className="transition-all duration-300 border-cosmic-purple/30 text-cosmic-purple hover:border-cosmic-purple hover:bg-cosmic-purple/10"
            >
              <Link href="/login">Log In</Link>
            </Button>
            <Button asChild size="sm" className="cosmic-button transition-all duration-300">
              <Link href="/register">Sign Up</Link>
            </Button>
          </div>
          <Button asChild size="sm" className="md:hidden cosmic-button transition-all duration-300">
            <Link href="/login">Log In</Link>
          </Button>

          <Button
            variant="ghost"
            size="icon"
            className="md:hidden text-cosmic-purple hover:text-cosmic-cyan hover:bg-cosmic-purple/10"
            onClick={toggleMobileMenu}
          >
            <Menu className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            className="fixed inset-0 z-50 bg-background/80 backdrop-blur-md md:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <motion.div
              className="flex h-full flex-col overflow-y-auto cosmic-glass py-6 shadow-lg"
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
            >
              <div className="px-6 py-4 flex justify-between items-center border-b border-cosmic-purple/20">
                <Link href="/" className="flex items-center gap-2" onClick={() => setIsMobileMenuOpen(false)}>
                  <GraduationCap className="h-6 w-6 text-cosmic-purple" />
                  <span className="text-xl font-bold text-cosmic-purple">GradeUP</span>
                </Link>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={toggleMobileMenu}
                  className="text-cosmic-purple hover:text-cosmic-cyan"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="lucide lucide-x"
                  >
                    <path d="M18 6 6 18" />
                    <path d="m6 6 12 12" />
                  </svg>
                </Button>
              </div>
              <div className="flex-1 px-6 py-4">
                <nav className="flex flex-col gap-4">
                  {navLinks.map((link) => (
                    <Link
                      key={link.href}
                      href={link.href}
                      className={`text-lg font-medium transition-colors hover:text-cosmic-cyan ${
                        pathname === link.href || (link.href !== "/" && pathname.startsWith(link.href))
                          ? "text-cosmic-purple"
                          : "text-foreground"
                      }`}
                      onClick={() => setIsMobileMenuOpen(false)}
                    >
                      {link.label}
                    </Link>
                  ))}
                </nav>
              </div>
              <div className="border-t border-cosmic-purple/20 px-6 py-4">
                <div className="flex flex-col gap-3">
                  <Button
                    asChild
                    variant="outline"
                    className="w-full border-cosmic-purple/30 text-cosmic-purple hover:bg-cosmic-purple/10"
                  >
                    <Link href="/login" onClick={() => setIsMobileMenuOpen(false)}>
                      Log In
                    </Link>
                  </Button>
                  <Button asChild className="w-full cosmic-button">
                    <Link href="/register" onClick={() => setIsMobileMenuOpen(false)}>
                      Sign Up
                    </Link>
                  </Button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}
