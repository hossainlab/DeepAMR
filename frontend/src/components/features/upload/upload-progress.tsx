"use client";

import { CheckCircle, Loader2, Circle } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import type { UploadProgress } from "@/types";

interface UploadProgressDisplayProps {
  progress: UploadProgress;
}

const stages = [
  { key: "uploading", label: "Uploading" },
  { key: "validating", label: "Validating" },
  { key: "analyzing", label: "Analyzing" },
  { key: "predicting", label: "Predicting" },
  { key: "completed", label: "Completed" },
] as const;

export function UploadProgressDisplay({ progress }: UploadProgressDisplayProps) {
  const currentStageIndex = stages.findIndex((s) => s.key === progress.stage);

  return (
    <div className="space-y-6">
      {/* Overall Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-foreground font-medium">{progress.message}</span>
          <span className="text-muted">
            {Math.round(((currentStageIndex * 100 + progress.progress) / (stages.length * 100)) * 100)}%
          </span>
        </div>
        <Progress
          value={((currentStageIndex * 100 + progress.progress) / (stages.length * 100)) * 100}
          className="h-2"
        />
      </div>

      {/* Stage Indicators */}
      <div className="space-y-3">
        {stages.map((stage, index) => {
          const isCompleted = index < currentStageIndex;
          const isCurrent = index === currentStageIndex;
          const isPending = index > currentStageIndex;

          return (
            <div
              key={stage.key}
              className={cn(
                "flex items-center gap-3 p-3 rounded-lg transition-all duration-300",
                isCompleted && "bg-susceptible/10",
                isCurrent && "bg-primary/10",
                isPending && "opacity-50"
              )}
            >
              {/* Status Icon */}
              <div
                className={cn(
                  "w-8 h-8 rounded-full flex items-center justify-center transition-colors",
                  isCompleted && "bg-susceptible text-white",
                  isCurrent && "bg-primary text-white",
                  isPending && "bg-surface-hover text-muted"
                )}
              >
                {isCompleted ? (
                  <CheckCircle className="h-5 w-5" />
                ) : isCurrent ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Circle className="h-5 w-5" />
                )}
              </div>

              {/* Label */}
              <div className="flex-1">
                <p
                  className={cn(
                    "text-sm font-medium",
                    isCompleted && "text-susceptible",
                    isCurrent && "text-primary",
                    isPending && "text-muted"
                  )}
                >
                  {stage.label}
                </p>
                {isCurrent && (
                  <p className="text-xs text-muted mt-0.5">{progress.message}</p>
                )}
              </div>

              {/* Stage Progress */}
              {isCurrent && (
                <div className="text-sm font-medium text-primary">
                  {progress.progress}%
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
