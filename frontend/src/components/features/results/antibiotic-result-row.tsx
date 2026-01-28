import { ResistanceBadge } from "./resistance-badge";
import { ConfidenceMeter } from "./confidence-meter";
import { TableCell, TableRow } from "@/components/ui/table";
import type { AntibioticResult } from "@/types";

interface AntibioticResultRowProps {
  result: AntibioticResult;
}

export function AntibioticResultRow({ result }: AntibioticResultRowProps) {
  return (
    <TableRow>
      <TableCell className="font-medium text-foreground">
        {result.antibiotic}
      </TableCell>
      <TableCell className="text-muted">
        {result.class}
      </TableCell>
      <TableCell>
        <ResistanceBadge status={result.status} showLabel />
      </TableCell>
      <TableCell>
        <ConfidenceMeter value={result.confidence} size="sm" />
      </TableCell>
      <TableCell className="text-muted font-mono text-sm">
        {result.mic || "-"}
      </TableCell>
      <TableCell className="text-muted font-mono text-sm">
        {result.breakpoint || "-"}
      </TableCell>
    </TableRow>
  );
}
