"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import {
  RoundedBox,
  Html,
  ContactShadows,
  Float,
  MeshReflectorMaterial,
  Environment,
  Lightformer,
} from "@react-three/drei";
import { Suspense, useRef } from "react";
import * as THREE from "three";
import type { Prediction } from "@/lib/types";
import { teamColor } from "@/lib/teams";
import { driverAvatarDataUri, driverImage } from "@/lib/driverImages";
import { pct } from "@/lib/format";

const STEP_META = [
  { rank: 1, x: 0, h: 1.55, metal: "#E7C558", z: 0 },
  { rank: 2, x: -2.15, h: 1.14, metal: "#C9D1D9", z: 0 },
  { rank: 3, x: 2.15, h: 0.84, metal: "#CD7F32", z: 0 },
];

function Step({
  meta,
  pred,
}: {
  meta: (typeof STEP_META)[number];
  pred: Prediction;
}) {
  const accent = teamColor(pred.team);
  const img = driverImage(pred.driver) ?? driverAvatarDataUri(pred.driver, pred.team);
  return (
    <group position={[meta.x, 0, meta.z]}>
      {/* Podium block — brushed metal, dark carbon core reads via low emissive. */}
      <RoundedBox
        args={[1.72, meta.h, 1.42]}
        radius={0.03}
        smoothness={5}
        position={[0, meta.h / 2, 0]}
      >
        <meshStandardMaterial
          color={meta.metal}
          metalness={0.95}
          roughness={0.32}
          envMapIntensity={0.9}
        />
      </RoundedBox>
      {/* Team-colour accent spine along the base. */}
      <mesh position={[0, 0.09, 0.72]}>
        <planeGeometry args={[1.72, 0.1]} />
        <meshBasicMaterial color={accent} toneMapped={false} />
      </mesh>
      {/* Driver card — squared telemetry tag, no coloured glow. */}
      <Float speed={1.4} rotationIntensity={0.08} floatIntensity={0.22}>
        <Html center position={[0, meta.h + 0.92, 0]} distanceFactor={7} pointerEvents="none">
          <div className="flex w-[168px] flex-col items-center gap-2 text-center">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={img}
              alt={pred.driver}
              width={64}
              height={64}
              className="rounded-[4px]"
              style={{
                boxShadow: "0 12px 30px -10px rgba(0,0,0,0.85)",
                outline: "1px solid rgba(255,255,255,0.12)",
              }}
            />
            <div className="flex flex-col items-center">
              <span
                className="font-text text-[19px] font-bold leading-none tracking-tight text-white"
                style={{ textShadow: "0 2px 14px rgba(0,0,0,0.9)" }}
              >
                {pred.driver}
              </span>
              <span className="mt-1.5 flex items-center gap-1.5">
                <span
                  className="inline-block h-2.5 w-0.5"
                  style={{ background: accent }}
                />
                <span className="font-mono text-[9.5px] uppercase tracking-[0.18em] text-white/70">
                  {pred.team}
                </span>
              </span>
            </div>
            <span
              className="t-data text-[11px] font-semibold text-white/90"
              style={{ textShadow: "0 1px 8px rgba(0,0,0,0.9)" }}
            >
              {pct(pred.p_win, 1)} win
            </span>
          </div>
        </Html>
      </Float>
    </group>
  );
}

function Scene({ top3 }: { top3: Prediction[] }) {
  const group = useRef<THREE.Group>(null);
  useFrame((state) => {
    if (!group.current) return;
    const t = state.clock.getElapsedTime();
    // Slow rotational drift + a faint breathing bob — camera-feel, not spin.
    group.current.rotation.y = Math.sin(t * 0.16) * 0.24;
    group.current.position.y = -1.2 + Math.sin(t * 0.5) * 0.02;
  });

  return (
    <>
      {/* In-scene environment (no network fetch) so the metal steps carry real
          reflections: a broad soft key overhead + a rosso and a cool card. */}
      <Environment resolution={256} environmentIntensity={0.4}>
        <Lightformer intensity={2.2} position={[0, 5, -3]} scale={[12, 5, 1]} color="#fff4e8" />
        <Lightformer intensity={1.6} position={[-6, 1, 3]} scale={[3, 8, 1]} color="#DC0000" />
        <Lightformer intensity={0.9} position={[6, 1, 3]} scale={[3, 8, 1]} color="#a9c4ff" />
      </Environment>
      <ambientLight intensity={0.2} />
      {/* Key */}
      <spotLight
        position={[6, 10, 6]}
        angle={0.5}
        penumbra={0.9}
        intensity={2.6}
        castShadow
        color="#fff6ee"
      />
      {/* Rosso rim from behind-left — the Ferrari signature edge light. */}
      <spotLight position={[-8, 4, -5]} angle={0.7} penumbra={1} intensity={3} color="#DC0000" />
      {/* Cool fill so the metal reads dimensional. */}
      <pointLight position={[2, 2, 6]} intensity={0.5} color="#cfe0ff" />

      <group ref={group} position={[0, -1.2, 0]}>
        {STEP_META.map((m) => {
          const pred = top3[m.rank - 1];
          return pred ? <Step key={m.rank} meta={m} pred={pred} /> : null;
        })}

        <ContactShadows position={[0, 0.01, 0]} opacity={0.6} scale={13} blur={2.6} far={4} />
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
          <planeGeometry args={[44, 44]} />
          <MeshReflectorMaterial
            resolution={512}
            mixBlur={1.1}
            mixStrength={26}
            blur={[320, 110]}
            depthScale={1}
            minDepthThreshold={0.4}
            maxDepthThreshold={1.3}
            color="#0a0a0a"
            metalness={0.7}
            roughness={0.85}
            mirror={0.4}
          />
        </mesh>
      </group>
    </>
  );
}

export default function Podium3D({ top3 }: { top3: Prediction[] }) {
  return (
    <Canvas
      dpr={[1, 1.8]}
      shadows
      camera={{ position: [0, 1.15, 7.4], fov: 40 }}
      gl={{ antialias: true, alpha: true }}
    >
      <Suspense fallback={null}>
        <Scene top3={top3} />
      </Suspense>
    </Canvas>
  );
}
