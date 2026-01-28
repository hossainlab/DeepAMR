import { LucideIcon, TrendingUp, TrendingDown } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface StatsCardProps {
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon: LucideIcon;
  iconColor?: string;
  trend?: "up" | "down" | "neutral";
}

export function StatsCard({
  title,
  value,
  change,
  changeLabel,
  icon: Icon,
  iconColor = "text-primary",
  trend,
}: StatsCardProps) {
  return (
    <Card className="card-hover">
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <p className="text-sm font-medium text-muted">{title}</p>
            <p className="text-3xl font-bold text-foreground mt-2">{value}</p>
            {change !== undefined && (
              <div className="flex items-center gap-1 mt-2">
                {trend === "up" && <TrendingUp className="h-4 w-4 text-susceptible" />}
                {trend === "down" && <TrendingDown className="h-4 w-4 text-destructive" />}
                <span
                  className={cn(
                    "text-sm font-medium",
                    trend === "up" && "text-susceptible",
                    trend === "down" && "text-destructive",
                    trend === "neutral" && "text-muted"
                  )}
                >
                  {change > 0 ? "+" : ""}
                  {change}%
                </span>
                {changeLabel && <span className="text-sm text-muted">{changeLabel}</span>}
              </div>
            )}
          </div>
          <div className={cn("p-3 rounded-xl bg-primary/10", iconColor.includes("destructive") && "bg-destructive/10", iconColor.includes("susceptible") && "bg-susceptible/10")}>
            <Icon className={cn("h-6 w-6", iconColor)} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
