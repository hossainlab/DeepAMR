"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ChartDataPoint } from "@/types";

interface ResistancePieChartProps {
  data: ChartDataPoint[];
  title?: string;
}

export function ResistancePieChart({ data, title = "Resistance Overview" }: ResistancePieChartProps) {
  const COLORS = {
    Resistant: "#ef4444",
    Intermediate: "#eab308",
    Susceptible: "#22c55e",
  };

  const total = data.reduce((sum, item) => sum + item.value, 0);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={80}
                paddingAngle={2}
                dataKey="value"
                nameKey="name"
              >
                {data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.color || COLORS[entry.name as keyof typeof COLORS] || "#94a3b8"}
                    strokeWidth={0}
                  />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  border: "1px solid #1e293b",
                  borderRadius: "8px",
                  color: "#f1f5f9",
                }}
                formatter={(value) => {
                  const numValue = typeof value === 'number' ? value : 0;
                  return `${numValue} (${((numValue / total) * 100).toFixed(1)}%)`;
                }}
              />
              <Legend
                verticalAlign="bottom"
                height={36}
                formatter={(value) => (
                  <span style={{ color: "#94a3b8", fontSize: "14px" }}>{value}</span>
                )}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-border">
          {data.map((item) => (
            <div key={item.name} className="text-center">
              <div
                className="text-2xl font-bold"
                style={{ color: item.color || COLORS[item.name as keyof typeof COLORS] }}
              >
                {item.value}
              </div>
              <div className="text-xs text-muted">{item.name}</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
