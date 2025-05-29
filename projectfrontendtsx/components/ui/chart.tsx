"use client"

import * as React from "react"
import { motion } from "framer-motion"

interface ChartContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string
  description?: string
}

const ChartContainer = React.forwardRef<HTMLDivElement, ChartContainerProps>(
  ({ className, title, description, ...props }, ref) => {
    return (
      <div ref={ref} className="space-y-1" {...props}>
        <h3 className="text-lg font-semibold">{title}</h3>
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
      </div>
    )
  },
)
ChartContainer.displayName = "ChartContainer"

interface ChartBarsProps extends React.SVGProps<SVGSVGElement> {
  data: {
    name: string
    value: number
  }[]
  yAxisWidth?: number
  showAnimation?: boolean
  children: (props: {
    key: string
    value: number
    name: string
    index: number
    formattedValue: string
    bar: {
      x: number
      y: number
      width: number
      height: number
    }
  }) => React.ReactNode
}

const ChartBars = React.forwardRef<SVGSVGElement, ChartBarsProps>(
  ({ className, data, yAxisWidth = 0, showAnimation = false, children, ...props }, ref) => {
    const maxValue = Math.max(...data.map((item) => item.value))
    return (
      <svg
        ref={ref}
        className={className}
        viewBox={`0 0 100 ${maxValue > 0 ? 100 : 0}`}
        preserveAspectRatio="none"
        {...props}
      >
        {data.map((item, index) => {
          const barHeight = (item.value / maxValue) * 100
          const barY = 100 - barHeight
          const barWidth = (100 - yAxisWidth) / data.length
          const barX = index * barWidth + yAxisWidth
          const key = `bar-${index}`
          const formattedValue = item.value.toString()

          return children({
            key,
            value: item.value,
            name: item.name,
            index,
            formattedValue,
            bar: {
              x: barX,
              y: barY,
              width: barWidth,
              height: barHeight,
            },
          })
        })}
      </svg>
    )
  },
)
ChartBars.displayName = "ChartBars"

interface ChartBarProps extends React.SVGProps<SVGRectElement> {}

const ChartBar = React.forwardRef<SVGRectElement, ChartBarProps>(({ className, ...props }, ref) => {
  return (
    <motion.rect
      ref={ref}
      className={className}
      initial={{ height: 0, y: 100 }}
      animate={{ height: props.height, y: props.y }}
      transition={{ duration: 0.5, delay: (props.x as number) * 0.1 }}
      {...props}
    />
  )
})
ChartBar.displayName = "ChartBar"

export { ChartContainer, ChartBars, ChartBar }
