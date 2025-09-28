'use client';

import ThreeScene from './components/ThreeScene';

export default function Home() {
  // Example: You can pass a model URL to load a 3D model
  // const modelUrl = '/path/to/your/model.glb';
  
  return (
    <div className="w-full h-screen bg-red-300 p-10">
      {/* To load a model, uncomment and modify the line below: */}
      <ThreeScene modelUrl="/human_teeth.glb" />
    </div>
  );
}