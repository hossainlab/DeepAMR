import { cn } from "@/lib/utils";
import type { ResistanceStatus } from "@/types";

interface ResistanceBadgeProps {
  status: ResistanceStatus;
  showLabel?: boolean;
  size?: "sm" | "md" | "lg";
}

const statusConfig = {
  R: {
    label: "Resistant",
    shortLabel: "R",
    className: "bg-resistant/20 text-resistant border-resistant/30",
    dotClassName: "bg-resistant",
  },
  I: {
    label: "Intermediate",
    shortLabel: "I",
    className: "bg-intermediate/20 text-intermediate border-intermediate/30",
    dotClassName: "bg-intermediate",
  },
  S: {
    label: "Susceptible",
    shortLabel: "S",
    className: "bg-susceptible/20 text-susceptible border-susceptible/30",
    dotClassName: "bg-susceptible",
  },
};

const sizeConfig = {
  sm: "px-1.5 py-0.5 text-xs",
  md: "px-2 py-1 text-sm",
  lg: "px-3 py-1.5 text-base",
};

export function ResistanceBadge({
  status,
  showLabel = false,
  size = "md",
}: ResistanceBadgeProps) {
  const config = statusConfig[status];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border font-semibold",
        config.className,
        sizeConfig[size]
      )}
    >
      <span className={cn("w-2 h-2 rounded-full", config.dotClassName)} />
      {showLabel ? config.label : config.shortLabel}
    </span>
  );
}
