"use client";

import { BarChart3, TrendingUp, Users, Activity } from "lucide-react";
import { Header } from "@/components/layout/header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ResistancePieChart } from "@/components/charts/resistance-pie-chart";
import { TrendChart } from "@/components/charts/trend-chart";
import { resistanceOverviewData, timeSeriesData } from "@/data/mock-predictions";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const organismData = [
  { name: "M. tuberculosis", value: 45, color: "#14b8a6" },
  { name: "E. coli", value: 32, color: "#06b6d4" },
  { name: "S. aureus", value: 28, color: "#8b5cf6" },
  { name: "K. pneumoniae", value: 18, color: "#f59e0b" },
  { name: "P. aeruginosa", value: 12, color: "#ef4444" },
  { name: "Others", value: 21, color: "#64748b" },
];

const weeklyPredictions = [
  { day: "Mon", predictions: 12 },
  { day: "Tue", predictions: 18 },
  { day: "Wed", predictions: 15 },
  { day: "Thu", predictions: 22 },
  { day: "Fri", predictions: 19 },
  { day: "Sat", predictions: 8 },
  { day: "Sun", predictions: 6 },
];

export default function AnalyticsPage() {
  return (
    <div className="min-h-screen">
      <Header
        title="Analytics"
        subtitle="System-wide statistics and insights"
      />

      <div className="p-6 space-y-6">
        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-xl bg-primary/10">
                  <Activity className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <p className="text-sm text-muted">This Week</p>
                  <p className="text-2xl font-bold text-foreground">156</p>
                  <p className="text-xs text-susceptible">+23% vs last week</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-xl bg-destructive/10">
                  <TrendingUp className="h-6 w-6 text-destructive" />
                </div>
                <div>
                  <p className="text-sm text-muted">Resistance Rate</p>
                  <p className="text-2xl font-bold text-foreground">34%</p>
                  <p className="text-xs text-destructive">+5% vs last month</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-xl bg-susceptible/10">
                  <Users className="h-6 w-6 text-susceptible" />
                </div>
                <div>
                  <p className="text-sm text-muted">Active Users</p>
                  <p className="text-2xl font-bold text-foreground">24</p>
                  <p className="text-xs text-muted">7 day active</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-xl bg-accent/10">
                  <BarChart3 className="h-6 w-6 text-accent" />
                </div>
                <div>
                  <p className="text-sm text-muted">Avg. Processing</p>
                  <p className="text-2xl font-bold text-foreground">8.5m</p>
                  <p className="text-xs text-susceptible">-12% faster</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Charts Row 1 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ResistancePieChart
            data={resistanceOverviewData}
            title="Overall Resistance Distribution"
          />

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Predictions by Organism</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={organismData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                    <XAxis type="number" stroke="#64748b" fontSize={12} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      stroke="#64748b"
                      fontSize={12}
                      width={100}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        border: "1px solid #1e293b",
                        borderRadius: "8px",
                        color: "#f1f5f9",
                      }}
                    />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {organismData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Charts Row 2 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TrendChart data={timeSeriesData} title="Weekly Resistance Trends" />

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Daily Predictions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={weeklyPredictions}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="day"
                      stroke="#64748b"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      stroke="#64748b"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        border: "1px solid #1e293b",
                        borderRadius: "8px",
                        color: "#f1f5f9",
                      }}
                    />
                    <Bar dataKey="predictions" fill="#14b8a6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Top Antibiotics Resistance */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Top Resistance Patterns</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { antibiotic: "Isoniazid", organism: "M. tuberculosis", rate: 42, count: 67 },
                { antibiotic: "Rifampicin", organism: "M. tuberculosis", rate: 38, count: 61 },
                { antibiotic: "Ampicillin", organism: "E. coli", rate: 65, count: 52 },
                { antibiotic: "Oxacillin", organism: "S. aureus", rate: 48, count: 45 },
                { antibiotic: "Ciprofloxacin", organism: "E. coli", rate: 35, count: 38 },
              ].map((item, index) => (
                <div
                  key={index}
                  className="flex items-center gap-4 p-4 rounded-lg bg-background-secondary"
                >
                  <div className="flex-1">
                    <p className="font-medium text-foreground">{item.antibiotic}</p>
                    <p className="text-sm text-muted italic">{item.organism}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-bold text-destructive">{item.rate}%</p>
                    <p className="text-xs text-muted">{item.count} isolates</p>
                  </div>
                  <div className="w-32 h-2 bg-surface-hover rounded-full overflow-hidden">
                    <div
                      className="h-full bg-destructive rounded-full"
                      style={{ width: `${item.rate}%` }}
                    />
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
