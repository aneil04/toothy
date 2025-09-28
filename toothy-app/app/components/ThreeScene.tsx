'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import { Suspense, useMemo } from 'react';
import { Mesh, Material } from 'three';

// Component to load and display a GLTF model with individual part control
function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  
  // Extract all meshes from the scene with their materials and positions
  const meshes = useMemo(() => {
    const extractedMeshes: Array<{
      name: string;
      geometry: any;
      material: Material | Material[];
      position: [number, number, number];
      rotation: [number, number, number];
      scale: [number, number, number];
    }> = [];
    
    scene.traverse((child) => {
      if (child instanceof Mesh && child.geometry && child.material) {
        extractedMeshes.push({
          name: child.name || `Part_${extractedMeshes.length}`,
          geometry: child.geometry,
          material: child.material,
          position: [child.position.x, child.position.y, child.position.z],
          rotation: [child.rotation.x, child.rotation.y, child.rotation.z],
          scale: [child.scale.x, child.scale.y, child.scale.z]
        });
      }
    });
    
    console.log(`Found ${extractedMeshes.length} parts:`, extractedMeshes.map(m => m.name));
    return extractedMeshes;
  }, [scene]);
  
  return (
    <group position={[0, 1, 0]} scale={1}>
      {meshes.map((mesh, index) => (
        <mesh
          key={`${mesh.name}-${index}`}
          geometry={mesh.geometry}
          material={mesh.material}
          position={mesh.position}
          rotation={mesh.rotation}
          scale={mesh.scale}
          name={mesh.name}
        >
          <meshStandardMaterial color="white" roughness={0.4} metalness={0.1} />
        </mesh>
      ))}
    </group>
  );
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