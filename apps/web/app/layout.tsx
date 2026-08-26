import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "B3 Portfolio Lab",
  description: "ML signals + exact quadratic knapsack portfolio construction for B3 research.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="pt-BR">
      <body>{children}</body>
    </html>
  );
}
