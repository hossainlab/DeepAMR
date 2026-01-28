"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Activity, AlertTriangle, CheckCircle, Clock, Plus, FileText } from "lucide-react";
import { Header } from "@/components/layout/header";
import { StatsCard } from "@/components/features/dashboard/stats-card";
import { RecentPredictions } from "@/components/features/dashboard/recent-predictions";
import { ResistancePieChart } from "@/components/charts/resistance-pie-chart";
import { TrendChart } from "@/components/charts/trend-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { predictionService } from "@/services/predictions";
import { dashboardStats, resistanceOverviewData, timeSeriesData } from "@/data/mock-predictions";
import type { Prediction } from "@/types";

export default function DashboardPage() {
  const [recentPredictions, setRecentPredictions] = useState<Prediction[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      const predictions = await predictionService.getRecent(5);
      setRecentPredictions(predictions);
      setIsLoading(false);
    };
    loadData();
  }, []);

  return (
    <div className="min-h-screen">
      <Header
        title="Dashboard"
        subtitle="Welcome back! Here's an overview of your AMR predictions."
      />

      <div className="p-6 space-y-6">
        {/* Quick Actions */}
        <div className="flex flex-wrap gap-4">
          <Link href="/upload">
            <Button className="gap-2">
              <Plus className="h-4 w-4" />
              New Upload
            </Button>
          </Link>
          <Link href="/reports">
            <Button variant="outline" className="gap-2">
              <FileText className="h-4 w-4" />
              View Reports
            </Button>
          </Link>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatsCard
            title="Total Predictions"
            value={dashboardStats.totalPredictions}
            change={dashboardStats.weeklyChange.predictions}
            changeLabel="vs last week"
            icon={Activity}
            trend="up"
          />
          <StatsCard
            title="Resistant Cases"
            value={dashboardStats.resistantCount}
            change={dashboardStats.weeklyChange.resistant}
            changeLabel="vs last week"
            icon={AlertTriangle}
            iconColor="text-destructive"
            trend="down"
          />
          <StatsCard
            title="Susceptible Cases"
            value={dashboardStats.susceptibleCount}
            icon={CheckCircle}
            iconColor="text-susceptible"
          />
          <StatsCard
            title="Pending Analysis"
            value={dashboardStats.pendingCount}
            icon={Clock}
            iconColor="text-intermediate"
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Predictions */}
          <div className="lg:col-span-2">
            {isLoading ? (
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-center h-64">
                    <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
                  </div>
                </CardContent>
              </Card>
            ) : (
              <RecentPredictions predictions={recentPredictions} />
            )}
          </div>

          {/* Resistance Overview */}
          <div>
            <ResistancePieChart data={resistanceOverviewData} />
          </div>
        </div>

        {/* Trend Chart */}
        <div className="grid grid-cols-1 gap-6">
          <TrendChart data={timeSeriesData} title="Weekly Resistance Trends" />
        </div>
      </div>
    </div>
  );
}
