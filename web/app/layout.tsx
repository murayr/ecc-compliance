import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "ECC Assistant",
  description: "Ask questions about the NCA Essential Cybersecurity Controls (ECC-2:2024)",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
