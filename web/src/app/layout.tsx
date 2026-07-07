import type { Metadata } from "next";
import { Anton, Archivo, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

// Display — Anton: an ultra-condensed motorsport-poster grotesque. Reserved for
// the largest moments (race name, section titles, big tabulated numbers). This
// is the committed voice, not a synthetic-italic fallback.
const display = Anton({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-display",
  display: "swap",
});

// Text/UI — Archivo: a normal-width neo-grotesque. Carries body, driver names,
// controls. Pairs with Anton on a width/weight contrast axis.
const text = Archivo({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-text",
  display: "swap",
});

// Data — IBM Plex Mono: every number, label and technical readout. Tabular.
const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Scuderia Predict — Formula 1 Finishing-Order Model",
  description:
    "A LightGBM learning-to-rank model with Plackett-Luce Monte-Carlo simulation, predicting Formula 1 finishing orders. The full model runs live in your browser.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${text.variable} ${mono.variable}`}
    >
      <body className="bg-ink font-text text-chalk antialiased">{children}</body>
    </html>
  );
}
