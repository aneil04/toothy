"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import ThreeScene from "../components/ThreeScene"

export default function VideoStreamPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [time, setTime] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null

    if (isRunning) {
      interval = setInterval(() => {
        setTime((prevTime) => prevTime + 1)
      }, 1000)
    } else if (!isRunning && time !== 0) {
      if (interval) clearInterval(interval)
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isRunning, time])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const handleStartStop = () => {
    setIsRunning(!isRunning)
  }

  const handleReset = () => {
    setTime(0)
    setIsRunning(false)
  }

  return (
    <div className="flex gap-8 mx-auto bg-gray-200 h-screen p-8">
      {/* Video Stream Area - Left Side */}
      <div className="flex-1">
        <div className="bg-gray-200 rounded-3xl h-full border-black flex items-center justify-center shadow-lg">
          <div className="text-4xl text-gray-600 font-light">video here</div>
        </div>
      </div>

      {/* Mobile Device Mockup - Right Side */}
      <div className="flex relative items-center bg-red-500 aspect-9/16 rounded-[3rem] shadow-lg">
          {/* Phone notch */}
          <div className="absolute top-6 left-1/2 transform -translate-x-1/2 bg-black rounded-full w-32 h-6 z-10"></div>

          {/* Phone screen with full canvas */}
          <div className="bg-white rounded-[2.5rem] border-8 border-black w-full h-full relative overflow-hidden">
            {/* <canvas ref={canvasRef} className="w-full h-full bg-red-200" width={320} height={640} /> */}
            <ThreeScene modelUrl="/teeth.glb" />
            <div className="absolute top-20 left-0 right-0 text-center z-20">
              <div className="text-6xl text-black rounded-lg mx-6 py-2">
                {formatTime(time)}
              </div>
            </div>

            <div className="absolute bottom-20 left-0 right-0 flex gap-3 justify-center items-center z-20">
              <Button
                onClick={handleStartStop}
              className="bg-blue-500 hover:bg-blue-600 border border-blue-500 hover:border-blue-600 text-white px-20 py-6 rounded-xl text-lg backdrop-blur-sm shadow-lg"
              >
                {isRunning ? "stop" : "start"}
              </Button>
              {time > 0 && (
                <Button
                  onClick={handleReset}
                  variant="outline"
                  className="px-20 py-6 rounded-xl text-lg bg-white/90 backdrop-blur-sm shadow-lg"
                >
                  reset
                </Button>
              )}
            </div>
          </div>
      </div>
    </div>
  )
}
