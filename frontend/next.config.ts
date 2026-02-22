import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // In development, proxy /api to local backend.
  // In production (Vercel), NEXT_PUBLIC_API_URL points directly to the backend,
  // so no rewrite is needed.
  async rewrites() {
    if (process.env.NEXT_PUBLIC_API_URL) {
      return [];
    }
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/:path*',
      },
    ];
  },
  // Allow images from any domain (for future use)
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },
};

export default nextConfig;
