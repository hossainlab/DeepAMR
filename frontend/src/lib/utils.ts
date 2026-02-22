import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(date: Date | string): string {
  const d = typeof date === "string" ? new Date(date) : date;
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function formatDateTime(date: Date | string): string {
  const d = typeof date === "string" ? new Date(date) : date;
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

export function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function getResistanceColor(status: "R" | "I" | "S"): string {
  switch (status) {
    case "R":
      return "var(--resistant)";
    case "I":
      return "var(--intermediate)";
    case "S":
      return "var(--susceptible)";
    default:
      return "var(--muted)";
  }
}

export function getResistanceLabel(status: "R" | "I" | "S"): string {
  switch (status) {
    case "R":
      return "Resistant";
    case "I":
      return "Intermediate";
    case "S":
      return "Susceptible";
    default:
      return "Unknown";
  }
}

// ---------------------------------------------------------------------------
// Report download helpers
// ---------------------------------------------------------------------------

import type { Prediction } from "@/types";

/** Trigger a file download in the browser. */
function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Download a single prediction as a CSV file. */
export function downloadPredictionCSV(prediction: Prediction) {
  const rows: string[][] = [
    ["DeepAMR Prediction Report"],
    [],
    ["Sample ID", prediction.sampleId],
    ["Organism", prediction.organism],
    ["Risk Level", prediction.overallRisk.toUpperCase()],
    ["Status", prediction.status],
    ["File", prediction.fileName || "N/A"],
    ["Created", prediction.createdAt ? formatDateTime(prediction.createdAt) : "N/A"],
    ["Completed", prediction.completedAt ? formatDateTime(prediction.completedAt) : "N/A"],
    [],
    ["Summary"],
    ["Resistant", String(prediction.summary?.resistant ?? 0)],
    ["Intermediate", String(prediction.summary?.intermediate ?? 0)],
    ["Susceptible", String(prediction.summary?.susceptible ?? 0)],
    [],
    ["Antibiotic", "Class", "Status", "Confidence"],
  ];

  if (prediction.results) {
    for (const r of prediction.results) {
      rows.push([
        r.antibiotic,
        r.class,
        getResistanceLabel(r.status),
        `${(r.confidence * 100).toFixed(1)}%`,
      ]);
    }
  }

  const csv = rows.map((r) => r.map((c) => `"${c}"`).join(",")).join("\n");
  downloadBlob(
    new Blob([csv], { type: "text/csv;charset=utf-8;" }),
    `DeepAMR_Report_${prediction.sampleId}.csv`,
  );
}

/** Download multiple predictions as a single CSV file. */
export function downloadBulkCSV(predictions: Prediction[]) {
  const header = [
    "Sample ID",
    "Organism",
    "Risk Level",
    "Resistant",
    "Intermediate",
    "Susceptible",
    "Status",
    "File",
    "Date",
  ];
  const rows = predictions.map((p) => [
    p.sampleId,
    p.organism,
    p.overallRisk.toUpperCase(),
    String(p.summary?.resistant ?? 0),
    String(p.summary?.intermediate ?? 0),
    String(p.summary?.susceptible ?? 0),
    p.status,
    p.fileName || "N/A",
    p.completedAt ? formatDateTime(p.completedAt) : formatDateTime(p.createdAt),
  ]);

  const csv = [header, ...rows].map((r) => r.map((c) => `"${c}"`).join(",")).join("\n");
  downloadBlob(
    new Blob([csv], { type: "text/csv;charset=utf-8;" }),
    `DeepAMR_Reports_${new Date().toISOString().slice(0, 10)}.csv`,
  );
}

/** Download a single prediction as a plain-text report (printable/PDF-friendly). */
export function downloadPredictionTextReport(prediction: Prediction) {
  const line = "=".repeat(60);
  const divider = "-".repeat(60);
  const parts: string[] = [
    line,
    "  DeepAMR - Antimicrobial Resistance Prediction Report",
    line,
    "",
    `Sample ID    : ${prediction.sampleId}`,
    `Organism     : ${prediction.organism}`,
    `Risk Level   : ${prediction.overallRisk.toUpperCase()}`,
    `Status       : ${prediction.status}`,
    `File         : ${prediction.fileName || "N/A"}`,
    `Created      : ${prediction.createdAt ? formatDateTime(prediction.createdAt) : "N/A"}`,
    `Completed    : ${prediction.completedAt ? formatDateTime(prediction.completedAt) : "N/A"}`,
    "",
    divider,
    "  Summary",
    divider,
    `  Resistant    : ${prediction.summary?.resistant ?? 0}`,
    `  Intermediate : ${prediction.summary?.intermediate ?? 0}`,
    `  Susceptible  : ${prediction.summary?.susceptible ?? 0}`,
    "",
    divider,
    "  Antibiotic Results",
    divider,
  ];

  if (prediction.results && prediction.results.length > 0) {
    const col = (s: string, w: number) => s.padEnd(w);
    parts.push(
      `  ${col("Antibiotic", 22)} ${col("Class", 18)} ${col("Status", 14)} Confidence`,
    );
    parts.push(`  ${"-".repeat(56)}`);
    for (const r of prediction.results) {
      parts.push(
        `  ${col(r.antibiotic, 22)} ${col(r.class, 18)} ${col(getResistanceLabel(r.status), 14)} ${(r.confidence * 100).toFixed(1)}%`,
      );
    }
  } else {
    parts.push("  No results available.");
  }

  parts.push("", line, `  Generated: ${formatDateTime(new Date())}`, line, "");

  downloadBlob(
    new Blob([parts.join("\n")], { type: "text/plain;charset=utf-8;" }),
    `DeepAMR_Report_${prediction.sampleId}.txt`,
  );
}
