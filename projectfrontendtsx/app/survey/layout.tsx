import type React from "react"

export default function SurveyLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return <div className="w-full h-full">{children}</div>
}
