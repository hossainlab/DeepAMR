"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Dna,
  LayoutDashboard,
  Upload,
  FileText,
  FileBarChart2,
  Settings,
  Users,
  BarChart3,
  LogOut,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/hooks/use-auth";

const mainNavItems = [
  { href: "/dashboard", icon: LayoutDashboard, label: "Dashboard" },
  { href: "/upload", icon: Upload, label: "Upload Sample" },
  { href: "/results", icon: FileText, label: "Results" },
  { href: "/reports", icon: FileBarChart2, label: "Reports" },
];

const adminNavItems = [
  { href: "/admin", icon: Settings, label: "Admin Overview" },
  { href: "/admin/users", icon: Users, label: "User Management" },
  { href: "/admin/analytics", icon: BarChart3, label: "Analytics" },
];

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout } = useAuth();
  const [isCollapsed, setIsCollapsed] = useState(false);

  const isAdmin = user?.role === "admin";

  const handleLogout = async () => {
    await logout();
    router.push("/");
  };

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 h-screen bg-surface border-r border-border transition-all duration-300",
        isCollapsed ? "w-16" : "w-64"
      )}
    >
      <div className="flex flex-col h-full">
        {/* Logo */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-border">
          <Link href="/dashboard" className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-primary/10">
              <Dna className="h-5 w-5 text-primary" />
            </div>
            {!isCollapsed && <span className="text-lg font-bold gradient-text">DeepAMR</span>}
          </Link>
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-1.5 rounded-md hover:bg-surface-hover text-muted hover:text-foreground transition-colors"
          >
            {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {mainNavItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all",
                pathname === item.href || pathname.startsWith(item.href + "/")
                  ? "bg-primary/10 text-primary"
                  : "text-muted hover:bg-surface-hover hover:text-foreground"
              )}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" />
              {!isCollapsed && <span>{item.label}</span>}
            </Link>
          ))}

          {isAdmin && (
            <>
              <Separator className="my-4" />
              <div className={cn("px-3 mb-2", isCollapsed && "hidden")}>
                <span className="text-xs font-semibold text-muted uppercase tracking-wider">
                  Admin
                </span>
              </div>
              {adminNavItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all",
                    pathname === item.href
                      ? "bg-primary/10 text-primary"
                      : "text-muted hover:bg-surface-hover hover:text-foreground"
                  )}
                >
                  <item.icon className="h-5 w-5 flex-shrink-0" />
                  {!isCollapsed && <span>{item.label}</span>}
                </Link>
              ))}
            </>
          )}
        </nav>

        {/* User Section */}
        <div className="p-3 border-t border-border">
          {!isCollapsed && user && (
            <div className="px-3 py-2 mb-2">
              <p className="text-sm font-medium text-foreground truncate">{user.name}</p>
              <p className="text-xs text-muted truncate">{user.email}</p>
            </div>
          )}
          <Button
            variant="ghost"
            className={cn(
              "w-full justify-start gap-3 text-muted hover:text-destructive hover:bg-destructive/10",
              isCollapsed && "justify-center px-0"
            )}
            onClick={handleLogout}
          >
            <LogOut className="h-5 w-5 flex-shrink-0" />
            {!isCollapsed && <span>Sign Out</span>}
          </Button>
        </div>
      </div>
    </aside>
  );
}
