'use client';

import ThreeScene from './components/ThreeScene';
import { useEffect, useRef, useState } from "react";

export default function Home() {
  // Example: You can pass a model URL to load a 3D model
  // const modelUrl = '/path/to/your/model.glb';

  return (
    <div className="w-full h-screen bg-red-300 p-10">
      {/* To load a model, uncomment and modify the line below: */}
      <ThreeScene modelUrl="/teeth.glb" />
      <WebcamCapture />
    </div>
  );
}

export function WebcamCapture() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const sendingRef = useRef(false);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const enableWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 }, // you can adjust resolution
          audio: false, // disable audio if not needed
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    };

    enableWebcam();
  }, []);

  // Capture frames on an interval
  useEffect(() => {
    const offscreenCanvas = document.createElement("canvas");
    let frameCount = Infinity

    const loop = async () => {
      if (!videoRef.current || !offscreenCanvas) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      if (sendingRef.current) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      sendingRef.current = true;
      frameCount++;
      try {
        const ctx = offscreenCanvas.getContext("2d");
        if (!ctx) return;

        offscreenCanvas.width = videoRef.current.videoWidth;
        offscreenCanvas.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

        const frameBlob = await new Promise<Blob>((resolve) =>
          (offscreenCanvas).toBlob(
            (b) => resolve(b!),
            "image/jpeg",
            0.7
          )
        );

        if (!frameBlob) {
          sendingRef.current = false;
        } else {
          // send blob to backend
          const form = new FormData();
          form.append("image", frameBlob, "frame.jpg");

          if (frameCount > 10) {
            frameCount = 0
            console.log("inference...");
            fetch("http://localhost:8000/infer", { method: "POST", body: form }).then(async (res) => {
              if (res.ok) {
                // Expect JSON from FastAPI: { detections: [...], latency_ms: ... }
                const json = await res.json();
                console.log("Inference response:", json);
              } else {
                console.warn("Inference HTTP error:", res.status);
              }
            });
          }

          const r = await fetch("http://localhost:8000/detect", { method: "POST", body: form });
          if (r.ok) {
            // Expect JSON from FastAPI: { detections: [...], latency_ms: ... }
            const blob = await r.blob();
            // Clean last URL to avoid memory leak
            setImgUrl((prev) => {
              if (prev) URL.revokeObjectURL(prev);
              return URL.createObjectURL(blob);
            });
            sendingRef.current = false;
          } else {
            console.warn("Inference HTTP error:", r.status);
          }
        }
      } catch (err) {
        console.error("Error capturing frame:", err);
      } finally {
        sendingRef.current = false;
      }

      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  return (
    <div>
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className='hidden'
      />
      {imgUrl && <img src={imgUrl} alt="Processed" className="rounded-3xl shadow-lg" />}
    </div>
  );
}
