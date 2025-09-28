'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF, OrbitControlsProps } from '@react-three/drei';
import { Suspense, useMemo, useRef, useEffect } from 'react';
import { Mesh, Material } from 'three';
import { CameraControls } from '@react-three/drei';

const gum_material = <meshStandardMaterial color="#fa7890" roughness={1} metalness={0} />
const tooth_material_50 = <meshStandardMaterial color="#eff6ff" roughness={1} metalness={0} />
const tooth_material_300 = <meshStandardMaterial color="#8ec5ff" roughness={1} metalness={0} />
const tooth_material_500 = <meshStandardMaterial color="#2b7fff" roughness={1} metalness={0} />

const material_map = [gum_material, tooth_material_50, tooth_material_300, tooth_material_500]
// Component to load and display a GLTF model with individual part control
function Model({ url, materials }: { url: string, materials: number[] }) {
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
          position={mesh.position}
          rotation={mesh.rotation}
          scale={mesh.scale}
          name={mesh.name}
        >
          {material_map[materials[index]]}
        </mesh>
      ))}
    </group>
  );
}

function Scene({ modelUrl, materials }: { modelUrl?: string, materials: number[] }) {
  return (
    <>
      {/* Basic lighting */}
      <ambientLight intensity={0.5} />

      {/* Load 3D model if URL provided, otherwise show placeholder */}
      {modelUrl && <Model url={modelUrl} materials={materials} />}

      {/* Environment for realistic lighting */}
      <Environment preset="sunset" />
    </>
  );
}

interface ThreeSceneProps {
  modelUrl?: string;
  materials: number[];
  target: number[];
}

export default function ThreeScene({ modelUrl, materials, target }: ThreeSceneProps) {
  const cameraControlRef = useRef<CameraControls | null>(null);

  useEffect(() => {
    cameraControlRef.current?.setPosition(target[0], target[1], target[2], true);
    cameraControlRef.current?.setTarget(0, 0, 0, true);
  }, [target]);

  return (
    <Canvas
      style={{ background: '#eff6ff' }}
      onClick={() => console.log(cameraControlRef.current?.camera.position)}
    >
      <Suspense fallback={null}>
        <CameraControls ref={cameraControlRef} />
        <Scene modelUrl={modelUrl} materials={materials} />
      </Suspense>
    </Canvas>
  );
}