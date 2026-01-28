import { cn } from "@/lib/utils";

interface ConfidenceMeterProps {
  value: number;
  showLabel?: boolean;
  size?: "sm" | "md" | "lg";
}

export function ConfidenceMeter({
  value,
  showLabel = true,
  size = "md",
}: ConfidenceMeterProps) {
  const percentage = Math.round(value * 100);

  const getColor = () => {
    if (percentage >= 90) return "bg-susceptible";
    if (percentage >= 70) return "bg-primary";
    if (percentage >= 50) return "bg-intermediate";
    return "bg-destructive";
  };

  const sizeConfig = {
    sm: { height: "h-1", width: "w-16", text: "text-xs" },
    md: { height: "h-2", width: "w-24", text: "text-sm" },
    lg: { height: "h-3", width: "w-32", text: "text-base" },
  };

  const config = sizeConfig[size];

  return (
    <div className="flex items-center gap-2">
      <div
        className={cn(
          "relative rounded-full bg-surface-hover overflow-hidden",
          config.height,
          config.width
        )}
      >
        <div
          className={cn("absolute left-0 top-0 h-full rounded-full transition-all", getColor())}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && (
        <span className={cn("text-muted font-medium", config.text)}>
          {percentage}%
        </span>
      )}
    </div>
  );
}
