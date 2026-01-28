"use client";

import { useEffect, useState } from "react";
import { Search, Filter, LayoutGrid, List } from "lucide-react";
import { Header } from "@/components/layout/header";
import { PredictionCard } from "@/components/features/results/prediction-card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { predictionService } from "@/services/predictions";
import type { Prediction, PredictionFilters, Organism } from "@/types";
import { cn } from "@/lib/utils";

export default function ResultsPage() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [filters, setFilters] = useState<PredictionFilters>({});
  const [searchTerm, setSearchTerm] = useState("");

  const organisms = predictionService.getOrganisms();

  useEffect(() => {
    const loadData = async () => {
      const data = await predictionService.getAll({
        ...filters,
        search: searchTerm,
      });
      setPredictions(data);
      setIsLoading(false);
    };
    loadData();
  }, [filters, searchTerm]);

  const clearFilters = () => {
    setFilters({});
    setSearchTerm("");
  };

  const hasActiveFilters = Object.keys(filters).length > 0 || searchTerm;

  return (
    <div className="min-h-screen">
      <Header
        title="Results"
        subtitle="View and manage your AMR prediction results"
      />

      <div className="p-6 space-y-6">
        {/* Filters Bar */}
        <div className="flex flex-wrap items-center gap-4">
          {/* Search */}
          <div className="relative flex-1 min-w-64">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted" />
            <Input
              placeholder="Search by sample ID, organism..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-9"
            />
          </div>

          {/* Organism Filter */}
          <Select
            value={filters.organism || "all"}
            onValueChange={(value) =>
              setFilters((prev) => ({
                ...prev,
                organism: value === "all" ? undefined : (value as Organism),
              }))
            }
          >
            <SelectTrigger className="w-52">
              <SelectValue placeholder="All Organisms" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Organisms</SelectItem>
              {organisms.map((organism) => (
                <SelectItem key={organism} value={organism}>
                  {organism}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Status Filter */}
          <Select
            value={filters.status || "all"}
            onValueChange={(value) =>
              setFilters((prev) => ({
                ...prev,
                status: value === "all" ? undefined : (value as Prediction["status"]),
              }))
            }
          >
            <SelectTrigger className="w-40">
              <SelectValue placeholder="All Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="processing">Processing</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>

          {/* Risk Filter */}
          <Select
            value={filters.risk || "all"}
            onValueChange={(value) =>
              setFilters((prev) => ({
                ...prev,
                risk: value === "all" ? undefined : (value as Prediction["overallRisk"]),
              }))
            }
          >
            <SelectTrigger className="w-40">
              <SelectValue placeholder="All Risk" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Risk Levels</SelectItem>
              <SelectItem value="high">High Risk</SelectItem>
              <SelectItem value="moderate">Moderate Risk</SelectItem>
              <SelectItem value="low">Low Risk</SelectItem>
            </SelectContent>
          </Select>

          {/* Clear Filters */}
          {hasActiveFilters && (
            <Button variant="ghost" size="sm" onClick={clearFilters}>
              Clear Filters
            </Button>
          )}

          {/* View Toggle */}
          <div className="flex items-center gap-1 p-1 rounded-lg bg-surface-hover">
            <Button
              variant={viewMode === "grid" ? "secondary" : "ghost"}
              size="icon"
              className="h-8 w-8"
              onClick={() => setViewMode("grid")}
            >
              <LayoutGrid className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === "list" ? "secondary" : "ghost"}
              size="icon"
              className="h-8 w-8"
              onClick={() => setViewMode("list")}
            >
              <List className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Results Count */}
        <div className="flex items-center gap-2">
          <Badge variant="secondary">{predictions.length} results</Badge>
          {hasActiveFilters && (
            <span className="text-sm text-muted">with active filters</span>
          )}
        </div>

        {/* Results Grid/List */}
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          </div>
        ) : predictions.length === 0 ? (
          <div className="text-center py-16">
            <Filter className="h-12 w-12 mx-auto mb-4 text-muted opacity-50" />
            <h3 className="text-lg font-semibold text-foreground mb-2">No results found</h3>
            <p className="text-muted mb-4">Try adjusting your filters or search term</p>
            {hasActiveFilters && (
              <Button variant="outline" onClick={clearFilters}>
                Clear Filters
              </Button>
            )}
          </div>
        ) : (
          <div
            className={cn(
              viewMode === "grid"
                ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
                : "space-y-4"
            )}
          >
            {predictions.map((prediction) => (
              <PredictionCard key={prediction.id} prediction={prediction} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
