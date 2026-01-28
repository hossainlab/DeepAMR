"use client";

import { useEffect } from "react";
import { Sidebar } from "@/components/layout/sidebar";
import { useAuth } from "@/hooks/use-auth";
import { currentUser } from "@/data/mock-users";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { user, isLoading, setUser } = useAuth();

  useEffect(() => {
    // For demo purposes, auto-login with demo user if not authenticated
    if (!isLoading && !user) {
      // Check localStorage first
      const stored = localStorage.getItem("user");
      if (stored) {
        setUser(JSON.parse(stored));
      } else {
        // Auto-login with demo user for convenience
        setUser(currentUser);
        localStorage.setItem("user", JSON.stringify(currentUser));
      }
    }
  }, [isLoading, user, setUser]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-muted">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <main className="pl-64 min-h-screen transition-all duration-300">
        {children}
      </main>
    </div>
  );
}
