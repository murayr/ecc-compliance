import type { ReactNode } from "react";
import { IBM_Plex_Mono, IBM_Plex_Sans, IBM_Plex_Sans_Arabic } from "next/font/google";
import "./globals.css";

// IBM Plex Sans Arabic and IBM Plex Sans were designed as one family, so Arabic and Latin text
// share x-height, weight and rhythm when mixed in one line. next/font self-hosts them at build time.
const arabic = IBM_Plex_Sans_Arabic({
  subsets: ["arabic"],
  weight: ["400", "600"],
  variable: "--font-ar",
  display: "swap",
});
const latin = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "600"],
  variable: "--font-en",
  display: "swap",
});
const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["500"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata = {
  title: "مساعد الضوابط الأساسية للأمن السيبراني · ECC Assistant",
  description: "اسأل عن الضوابط الأساسية للأمن السيبراني (ECC-2:2024) بالعربية أو الإنجليزية",
};

export const viewport = {
  themeColor: "#040806",
  colorScheme: "dark",
};

// Arabic first: the document is Arabic and right-to-left; English content opts into LTR where it appears.
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ar" dir="rtl" className={`${arabic.variable} ${latin.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
