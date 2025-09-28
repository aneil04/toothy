"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import ThreeScene from "./components/ThreeScene"
import WebcamCapture from "./components/WebcamCapture"

const name2material_index = {
  "closed_left": 0,
  "closed_mid": 1,
  "closed_right": 2,
  "open_left_down": 4,
  "open_mid_down": 6,
  "open_right_down": 8,
  "open_left_up": 5,
  "open_mid_up": 7,
  "open_right_up": 9,
}

export default function VideoStreamPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [time, setTime] = useState(0)
  const [inferenceResult, setInferenceResult] = useState<any>(null)
  const [materials, setMaterials] = useState<number[]>([1, 1, 1, 0, 1, 1, 1, 1, 1, 1])

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

  const incrementMaterial = (index: number) => {
    setMaterials((prevMaterials) => {
      const newMaterials = [...prevMaterials]
      newMaterials[index] = Math.min((newMaterials[index] + 1), 3)
      return newMaterials
    })
  }
  
  useEffect(() => {
    if (!inferenceResult) {
      return
    }

    const section = inferenceResult.pred_name
    const index = name2material_index[section as keyof typeof name2material_index]
    incrementMaterial(index)
  }, [inferenceResult])

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
    setMaterials([1, 1, 1, 0, 1, 1, 1, 1, 1, 1])
    setInferenceResult(null)
  }

  return (
    <div className="flex flex-row gap-20 justify-center bg-gray-200 h-screen p-20">
      {/* Video Stream Area - Left Side */}
      <div className="flex flex-col gap-8">
        <WebcamCapture setInferenceResult={setInferenceResult} />
        <div className="flex flex-col gap-2">
          <span className="text-4xl text-gray-600 font-light">{inferenceResult ? `Prediction: ${inferenceResult.pred_name}` : "inference..."}</span>
        </div>
      </div>

      {/* Mobile Device Mockup - Right Side */}
      <div className="flex relative items-center bg-red-500 aspect-[9/18] rounded-[3rem] shadow-lg">
        {/* Phone notch */}
        <div className="absolute top-6 left-1/2 transform -translate-x-1/2 bg-black rounded-full w-32 h-6 z-10"></div>

        {/* Phone screen with full canvas */}
        <div className="bg-white rounded-[2.5rem] border-8 border-black w-full h-full relative overflow-hidden">
          {/* <canvas ref={canvasRef} className="w-full h-full bg-red-200" width={320} height={640} /> */}
          <ThreeScene modelUrl="/teeth.glb" materials={materials} />
          <div className="absolute top-20 left-0 right-0 text-center z-20">
            <div className="text-6xl text-black rounded-lg mx-6 py-2">
              {formatTime(time)}
            </div>
          </div>

          <div className="absolute bottom-20 left-0 right-0 flex gap-3 justify-center items-center z-20">
            <Button
              onClick={handleStartStop}
              className="bg-blue-500 hover:bg-blue-600 border border-blue-500 hover:border-blue-600 text-white px-10 py-6 rounded-xl text-lg backdrop-blur-sm shadow-lg"
            >
              {isRunning ? "stop" : "start"}
            </Button>
            {time > 0 && (
              <Button
                onClick={handleReset}
                variant="outline"
                className="px-10 py-6 rounded-xl text-lg bg-white/90 backdrop-blur-sm shadow-lg border-gray-300 border-2"
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
