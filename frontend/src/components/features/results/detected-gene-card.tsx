import { Dna, Info } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { DetectedGene } from "@/types";

interface DetectedGeneCardProps {
  gene: DetectedGene;
}

export function DetectedGeneCard({ gene }: DetectedGeneCardProps) {
  return (
    <Card className="card-hover">
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-primary/10">
              <Dna className="h-4 w-4 text-primary" />
            </div>
            <div>
              <h4 className="font-semibold text-foreground font-mono">{gene.gene}</h4>
              <p className="text-xs text-muted">{gene.mechanism}</p>
            </div>
          </div>
          <Tooltip>
            <TooltipTrigger>
              <Info className="h-4 w-4 text-muted hover:text-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              <p>{gene.description}</p>
            </TooltipContent>
          </Tooltip>
        </div>

        {/* Coverage & Identity */}
        <div className="space-y-2 mb-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted">Coverage</span>
            <span className="text-foreground">{gene.coverage}%</span>
          </div>
          <Progress value={gene.coverage} className="h-1" />

          <div className="flex items-center justify-between text-xs">
            <span className="text-muted">Identity</span>
            <span className="text-foreground">{gene.identity}%</span>
          </div>
          <Progress value={gene.identity} className="h-1" />
        </div>

        {/* Associated Antibiotics */}
        <div className="flex flex-wrap gap-1">
          {gene.antibiotics.map((antibiotic) => (
            <Badge key={antibiotic} variant="secondary" className="text-xs">
              {antibiotic}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
