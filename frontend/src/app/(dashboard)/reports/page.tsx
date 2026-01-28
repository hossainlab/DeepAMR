"use client";

import { useEffect, useState } from "react";
import { Download, FileText, Search, Calendar } from "lucide-react";
import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { predictionService } from "@/services/predictions";
import { formatDateTime, formatFileSize } from "@/lib/utils";
import type { Prediction } from "@/types";

export default function ReportsPage() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());

  useEffect(() => {
    const loadData = async () => {
      const data = await predictionService.getAll({ status: "completed" });
      setPredictions(data);
      setIsLoading(false);
    };
    loadData();
  }, []);

  const filteredPredictions = predictions.filter(
    (p) =>
      p.sampleId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.organism.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const toggleRowSelection = (id: string) => {
    const newSelected = new Set(selectedRows);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedRows(newSelected);
  };

  const toggleAllRows = () => {
    if (selectedRows.size === filteredPredictions.length) {
      setSelectedRows(new Set());
    } else {
      setSelectedRows(new Set(filteredPredictions.map((p) => p.id)));
    }
  };

  const handleBulkDownload = () => {
    alert(`Downloading ${selectedRows.size} reports...`);
  };

  const handleSingleDownload = (prediction: Prediction) => {
    alert(`Downloading report for ${prediction.sampleId}...`);
  };

  const getRiskVariant = (risk: Prediction["overallRisk"]) => {
    switch (risk) {
      case "high":
        return "destructive";
      case "moderate":
        return "warning";
      case "low":
        return "success";
    }
  };

  return (
    <div className="min-h-screen">
      <Header
        title="Reports"
        subtitle="Download and manage prediction reports"
      />

      <div className="p-6 space-y-6">
        {/* Actions Bar */}
        <div className="flex flex-wrap items-center gap-4">
          {/* Search */}
          <div className="relative flex-1 min-w-64">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted" />
            <Input
              placeholder="Search reports..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-9"
            />
          </div>

          {/* Date Range */}
          <Select defaultValue="all">
            <SelectTrigger className="w-40">
              <Calendar className="h-4 w-4 mr-2" />
              <SelectValue placeholder="Date Range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Time</SelectItem>
              <SelectItem value="today">Today</SelectItem>
              <SelectItem value="week">This Week</SelectItem>
              <SelectItem value="month">This Month</SelectItem>
            </SelectContent>
          </Select>

          {/* Bulk Actions */}
          {selectedRows.size > 0 && (
            <Button onClick={handleBulkDownload} className="gap-2">
              <Download className="h-4 w-4" />
              Download Selected ({selectedRows.size})
            </Button>
          )}
        </div>

        {/* Reports Table */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              Completed Reports
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {isLoading ? (
              <div className="flex items-center justify-center h-64">
                <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
              </div>
            ) : filteredPredictions.length === 0 ? (
              <div className="text-center py-16">
                <FileText className="h-12 w-12 mx-auto mb-4 text-muted opacity-50" />
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  No reports found
                </h3>
                <p className="text-muted">
                  Completed predictions will appear here as downloadable reports
                </p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12">
                      <input
                        type="checkbox"
                        checked={selectedRows.size === filteredPredictions.length}
                        onChange={toggleAllRows}
                        className="rounded border-border"
                      />
                    </TableHead>
                    <TableHead>Sample ID</TableHead>
                    <TableHead>Organism</TableHead>
                    <TableHead>Risk Level</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead>File Size</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredPredictions.map((prediction) => (
                    <TableRow key={prediction.id}>
                      <TableCell>
                        <input
                          type="checkbox"
                          checked={selectedRows.has(prediction.id)}
                          onChange={() => toggleRowSelection(prediction.id)}
                          className="rounded border-border"
                        />
                      </TableCell>
                      <TableCell className="font-medium">
                        {prediction.sampleId}
                      </TableCell>
                      <TableCell className="text-muted italic">
                        {prediction.organism}
                      </TableCell>
                      <TableCell>
                        <Badge variant={getRiskVariant(prediction.overallRisk)}>
                          {prediction.overallRisk}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-muted">
                        {formatDateTime(prediction.completedAt || prediction.createdAt)}
                      </TableCell>
                      <TableCell className="text-muted">
                        {formatFileSize(prediction.fileSize)}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleSingleDownload(prediction)}
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
