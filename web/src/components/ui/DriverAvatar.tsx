"use client";

/* eslint-disable @next/next/no-img-element */
import { driverImage, driverAvatarDataUri } from "@/lib/driverImages";

export default function DriverAvatar({
  driver,
  team,
  size = 40,
  className = "",
}: {
  driver: string;
  team: string;
  size?: number;
  className?: string;
}) {
  const real = driverImage(driver);
  const src = real ?? driverAvatarDataUri(driver, team);
  return (
    <img
      src={src}
      alt={driver}
      width={size}
      height={size}
      style={{ width: size, height: size }}
      className={`shrink-0 rounded-[3px] object-cover ring-1 ring-white/10 ${className}`}
    />
  );
}
