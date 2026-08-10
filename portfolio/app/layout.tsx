import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

const title = "Toronto Airbnb Market Network — Interactive Case Study";
const description =
  "An interactive network-science case study of 15,809 Toronto Airbnb listings, community structure, robustness, and price influence.";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host =
    requestHeaders.get("x-forwarded-host") ??
    requestHeaders.get("host") ??
    "localhost:4173";
  const protocol =
    requestHeaders.get("x-forwarded-proto") ??
    (host.startsWith("localhost") || host.startsWith("127.0.0.1")
      ? "http"
      : "https");
  const origin = `${protocol}://${host}`;
  const imageUrl = `${origin}/og.png`;

  return {
    title,
    description,
    openGraph: {
      type: "website",
      url: origin,
      title,
      description,
      siteName: "Toronto Airbnb Market Network",
      images: [
        {
          url: imageUrl,
          width: 1735,
          height: 907,
          alt: "Toronto Airbnb Market Network — 15,809 listings and 17 market communities",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [imageUrl],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
