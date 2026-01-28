import Link from "next/link";
import { Clock, FileText, AlertTriangle, CheckCircle, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatDateTime } from "@/lib/utils";
import type { Prediction } from "@/types";

interface PredictionCardProps {
  prediction: Prediction;
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  const getStatusIcon = () => {
    switch (prediction.status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-susceptible" />;
      case "processing":
        return <Loader2 className="h-4 w-4 text-primary animate-spin" />;
      case "pending":
        return <Clock className="h-4 w-4 text-muted" />;
      case "failed":
        return <AlertTriangle className="h-4 w-4 text-destructive" />;
    }
  };

  const getStatusVariant = () => {
    switch (prediction.status) {
      case "completed":
        return "success";
      case "processing":
        return "default";
      case "pending":
        return "secondary";
      case "failed":
        return "destructive";
    }
  };

  const getRiskVariant = () => {
    switch (prediction.overallRisk) {
      case "high":
        return "destructive";
      case "moderate":
        return "warning";
      case "low":
        return "success";
    }
  };

  return (
    <Link href={`/results/${prediction.id}`}>
      <Card hover className="h-full">
        <CardContent className="p-5">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              <span className="font-semibold text-foreground">{prediction.sampleId}</span>
            </div>
            <Badge variant={getStatusVariant()}>
              <span className="flex items-center gap-1">
                {getStatusIcon()}
                {prediction.status}
              </span>
            </Badge>
          </div>

          {/* Organism */}
          <p className="text-sm text-muted mb-4 italic">{prediction.organism}</p>

          {/* Stats */}
          {prediction.status === "completed" && prediction.summary && (
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="text-center p-2 rounded-lg bg-destructive/10">
                <div className="text-lg font-bold text-destructive">
                  {prediction.summary.resistant}
                </div>
                <div className="text-xs text-muted">Resistant</div>
              </div>
              <div className="text-center p-2 rounded-lg bg-intermediate/10">
                <div className="text-lg font-bold text-intermediate">
                  {prediction.summary.intermediate}
                </div>
                <div className="text-xs text-muted">Intermediate</div>
              </div>
              <div className="text-center p-2 rounded-lg bg-susceptible/10">
                <div className="text-lg font-bold text-susceptible">
                  {prediction.summary.susceptible}
                </div>
                <div className="text-xs text-muted">Susceptible</div>
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="flex items-center justify-between pt-4 border-t border-border">
            <div className="flex items-center gap-1 text-xs text-muted">
              <Clock className="h-3 w-3" />
              {formatDateTime(prediction.createdAt)}
            </div>
            {prediction.status === "completed" && (
              <Badge variant={getRiskVariant()}>
                {prediction.overallRisk} risk
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
