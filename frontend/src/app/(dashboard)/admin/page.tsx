"use client";

import { useState } from "react";
import Link from "next/link";
import {
  Users,
  Activity,
  HardDrive,
  TrendingUp,
  ArrowRight,
  Clock,
} from "lucide-react";
import { Header } from "@/components/layout/header";
import { StatsCard } from "@/components/features/dashboard/stats-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { adminStats, recentActivity } from "@/data/mock-users";
import { formatDateTime } from "@/lib/utils";

export default function AdminPage() {
  const [stats] = useState(adminStats);
  const [activity] = useState(recentActivity);

  const storagePercentage = (stats.storageUsed / stats.storageLimit) * 100;

  return (
    <div className="min-h-screen">
      <Header
        title="Admin Overview"
        subtitle="System statistics and recent activity"
      />

      <div className="p-6 space-y-6">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatsCard
            title="Total Users"
            value={stats.totalUsers}
            icon={Users}
            trend="up"
            change={12}
            changeLabel="this month"
          />
          <StatsCard
            title="Active Users"
            value={stats.activeUsers}
            icon={Activity}
            iconColor="text-susceptible"
          />
          <StatsCard
            title="Total Predictions"
            value={stats.totalPredictions}
            icon={TrendingUp}
            trend="up"
            change={23}
            changeLabel="this week"
          />
          <StatsCard
            title="Predictions Today"
            value={stats.predictionsToday}
            icon={Clock}
            iconColor="text-accent"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Storage Usage */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <HardDrive className="h-5 w-5 text-primary" />
                Storage Usage
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-muted">Used</span>
                <span className="font-medium">
                  {stats.storageUsed} GB / {stats.storageLimit} GB
                </span>
              </div>
              <Progress value={storagePercentage} className="h-3" />
              <p className="text-sm text-muted">
                {(stats.storageLimit - stats.storageUsed).toFixed(1)} GB available
              </p>
            </CardContent>
          </Card>

          {/* Quick Links */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Link href="/admin/users">
                <Button variant="outline" className="w-full justify-between">
                  Manage Users
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link href="/admin/analytics">
                <Button variant="outline" className="w-full justify-between">
                  View Analytics
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link href="/upload">
                <Button variant="outline" className="w-full justify-between">
                  New Prediction
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Recent Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {activity.map((item, index) => (
                <div
                  key={index}
                  className="flex items-center gap-4 p-3 rounded-lg bg-background-secondary"
                >
                  <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary font-medium">
                    {item.userName
                      .split(" ")
                      .map((n) => n[0])
                      .join("")}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-foreground">
                      {item.userName}
                    </p>
                    <p className="text-sm text-muted">
                      {item.action}
                      {item.details && (
                        <span className="text-primary ml-1">{item.details}</span>
                      )}
                    </p>
                  </div>
                  <div className="text-xs text-muted">
                    {formatDateTime(item.timestamp)}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
