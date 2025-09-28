'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import { Suspense } from 'react';

// Component to load and display a GLTF model
function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  return <primitive object={scene} scale={1} position={[0, 1, 0]} />;
}


function Scene({ modelUrl }: { modelUrl?: string }) {
  return (
    <>
      {/* Basic lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />

      {/* Load 3D model if URL provided, otherwise show placeholder */}
      {modelUrl && <Model url={modelUrl} />}

      {/* Environment for realistic lighting */}
      <Environment preset="sunset" />
    </>
  );
}

interface ThreeSceneProps {
  modelUrl?: string;
}

export default function ThreeScene({ modelUrl }: ThreeSceneProps) {
  return (
    <Canvas
      camera={{ position: [0, 0, 5], fov: 75 }}
      style={{ background: '#f0f0f0' }}
    >
      <Suspense fallback={null}>
        <Scene modelUrl={modelUrl} />
        <OrbitControls 
          enablePan={true} 
          enableZoom={true} 
          enableRotate={true}
          autoRotate={true}
          autoRotateSpeed={0.5}
        />
      </Suspense>
    </Canvas>
  );
}