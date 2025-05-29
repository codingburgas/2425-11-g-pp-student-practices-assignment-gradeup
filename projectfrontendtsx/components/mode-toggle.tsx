"use client"

import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { useEffect, useState } from "react"

export function ModeToggle() {
  const { setTheme, theme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <Button variant="outline" size="icon" className="h-9 w-9">
        <Sun className="h-[1.2rem] w-[1.2rem]" />
        <span className="sr-only">Toggle theme</span>
      </Button>
    )
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="icon"
          className="h-9 w-9 relative overflow-hidden group border-cosmic-purple/30 hover:border-cosmic-purple"
        >
          <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all duration-300 dark:-rotate-90 dark:scale-0 text-cosmic-purple" />
          <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all duration-300 dark:rotate-0 dark:scale-100 text-cosmic-cyan" />
          <span className="sr-only">Toggle theme</span>
          <span className="absolute inset-0 rounded-md bg-cosmic-purple/0 transition-colors group-hover:bg-cosmic-purple/10"></span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="cosmic-glass border-cosmic-purple/30 animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95"
      >
        <DropdownMenuItem
          onClick={() => setTheme("light")}
          className="flex items-center gap-2 cursor-pointer hover:bg-cosmic-purple/20 focus:bg-cosmic-purple/20"
        >
          <Sun className="h-4 w-4 text-cosmic-purple" />
          <span>Light</span>
          {theme === "light" && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-cosmic-purple"></span>}
        </DropdownMenuItem>
        <DropdownMenuItem
          onClick={() => setTheme("dark")}
          className="flex items-center gap-2 cursor-pointer hover:bg-cosmic-purple/20 focus:bg-cosmic-purple/20"
        >
          <Moon className="h-4 w-4 text-cosmic-cyan" />
          <span>Dark</span>
          {theme === "dark" && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-cosmic-cyan"></span>}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
