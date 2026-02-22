"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Activity, AlertTriangle, CheckCircle, Clock, Plus, FileText, BarChart3 } from "lucide-react";
import { Header } from "@/components/layout/header";
import { StatsCard } from "@/components/features/dashboard/stats-card";
import { RecentPredictions } from "@/components/features/dashboard/recent-predictions";
import { ResistancePieChart } from "@/components/charts/resistance-pie-chart";
import { TrendChart } from "@/components/charts/trend-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { predictionService } from "@/services/predictions";
import { deepamrApi } from "@/services/api";
import type { ModelPerformance } from "@/services/api";
import type { Prediction, DashboardStats } from "@/types";

export default function DashboardPage() {
  const [recentPredictions, setRecentPredictions] = useState<Prediction[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [resistanceData, setResistanceData] = useState<Array<{ name: string; value: number; color: string }>>([]);
  const [trendsData, setTrendsData] = useState<Array<{ date: string; resistant: number; susceptible: number; intermediate: number }>>([]);
  const [modelPerf, setModelPerf] = useState<ModelPerformance | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      const [predictions, dashStats, resistance, trends, perf] = await Promise.all([
        predictionService.getRecent(5),
        deepamrApi.dashboard.getStats().catch(() => null),
        deepamrApi.dashboard.getResistanceOverview().catch(() => []),
        deepamrApi.dashboard.getTrends().catch(() => []),
        deepamrApi.getModelPerformance().catch(() => null),
      ]);
      setRecentPredictions(predictions);
      setStats(dashStats);
      setResistanceData(resistance);
      setTrendsData(trends);
      setModelPerf(perf);
      setIsLoading(false);
    };
    loadData();
  }, []);

  const defaultStats: DashboardStats = {
    totalPredictions: 0,
    resistantCount: 0,
    susceptibleCount: 0,
    pendingCount: 0,
    weeklyChange: { predictions: 0, resistant: 0 },
  };

  const s = stats || defaultStats;

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
            value={s.totalPredictions}
            change={s.weeklyChange.predictions}
            changeLabel="vs last week"
            icon={Activity}
            trend="up"
          />
          <StatsCard
            title="Resistant Cases"
            value={s.resistantCount}
            change={s.weeklyChange.resistant}
            changeLabel="vs last week"
            icon={AlertTriangle}
            iconColor="text-destructive"
            trend="down"
          />
          <StatsCard
            title="Susceptible Cases"
            value={s.susceptibleCount}
            icon={CheckCircle}
            iconColor="text-susceptible"
          />
          <StatsCard
            title="Pending Analysis"
            value={s.pendingCount}
            icon={Clock}
            iconColor="text-intermediate"
          />
        </div>

        {/* Model Performance */}
        {modelPerf && (
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="h-5 w-5 text-primary" />
                <h3 className="font-semibold text-lg">Model Performance</h3>
                <span className="text-xs text-muted ml-auto">v{modelPerf.model_version} — Advanced Deep Learning</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 rounded-lg bg-primary/5">
                  <div className="text-2xl font-bold text-primary">{(modelPerf.overall.micro_f1 * 100).toFixed(1)}%</div>
                  <div className="text-xs text-muted mt-1">Micro F1 Score</div>
                </div>
                <div className="text-center p-3 rounded-lg bg-primary/5">
                  <div className="text-2xl font-bold text-primary">{(modelPerf.overall.auc * 100).toFixed(1)}%</div>
                  <div className="text-xs text-muted mt-1">AUC-ROC</div>
                </div>
                <div className="text-center p-3 rounded-lg bg-primary/5">
                  <div className="text-2xl font-bold text-primary">{(modelPerf.overall.macro_f1 * 100).toFixed(1)}%</div>
                  <div className="text-xs text-muted mt-1">Macro F1 Score</div>
                </div>
                <div className="text-center p-3 rounded-lg bg-primary/5">
                  <div className="text-2xl font-bold text-primary">{(modelPerf.overall.hamming_loss * 100).toFixed(1)}%</div>
                  <div className="text-xs text-muted mt-1">Hamming Loss</div>
                </div>
              </div>
              <p className="text-xs text-muted mt-3">
                Evaluated on held-out test set. Results assist clinical decisions and do not replace laboratory testing.
              </p>
            </CardContent>
          </Card>
        )}

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
            <ResistancePieChart data={resistanceData} />
          </div>
        </div>

        {/* Trend Chart */}
        <div className="grid grid-cols-1 gap-6">
          <TrendChart data={trendsData} title="Weekly Resistance Trends" />
        </div>
      </div>
    </div>
  );
}
