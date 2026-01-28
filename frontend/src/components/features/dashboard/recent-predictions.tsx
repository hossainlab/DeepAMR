"use client";

import Link from "next/link";
import { ArrowRight, Clock, FileText } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatDateTime } from "@/lib/utils";
import type { Prediction } from "@/types";

interface RecentPredictionsProps {
  predictions: Prediction[];
}

export function RecentPredictions({ predictions }: RecentPredictionsProps) {
  const getStatusVariant = (status: Prediction["status"]) => {
    switch (status) {
      case "completed":
        return "success";
      case "processing":
        return "default";
      case "pending":
        return "secondary";
      case "failed":
        return "destructive";
      default:
        return "secondary";
    }
  };

  const getRiskVariant = (risk: Prediction["overallRisk"]) => {
    switch (risk) {
      case "high":
        return "destructive";
      case "moderate":
        return "warning";
      case "low":
        return "success";
      default:
        return "secondary";
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-lg flex items-center gap-2">
          <FileText className="h-5 w-5 text-primary" />
          Recent Predictions
        </CardTitle>
        <Link href="/results">
          <Button variant="ghost" size="sm" className="text-primary">
            View All
            <ArrowRight className="h-4 w-4 ml-1" />
          </Button>
        </Link>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {predictions.length === 0 ? (
            <div className="text-center py-8 text-muted">
              <FileText className="h-12 w-12 mx-auto mb-4 opacity-20" />
              <p>No predictions yet</p>
              <Link href="/upload">
                <Button variant="outline" size="sm" className="mt-4">
                  Upload First Sample
                </Button>
              </Link>
            </div>
          ) : (
            predictions.map((prediction) => (
              <Link
                key={prediction.id}
                href={`/results/${prediction.id}`}
                className="block"
              >
                <div className="flex items-center justify-between p-4 rounded-lg bg-background-secondary hover:bg-surface-hover transition-colors cursor-pointer">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium text-foreground truncate">
                        {prediction.sampleId}
                      </span>
                      <Badge variant={getStatusVariant(prediction.status)}>
                        {prediction.status}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-4 text-sm text-muted">
                      <span>{prediction.organism}</span>
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {formatDateTime(prediction.createdAt)}
                      </span>
                    </div>
                  </div>
                  <div className="ml-4">
                    {prediction.status === "completed" && (
                      <Badge variant={getRiskVariant(prediction.overallRisk)}>
                        {prediction.overallRisk} risk
                      </Badge>
                    )}
                  </div>
                </div>
              </Link>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}
