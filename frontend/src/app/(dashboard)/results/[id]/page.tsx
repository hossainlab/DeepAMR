"use client";

import { useEffect, useState, use } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Download,
  Dna,
  AlertTriangle,
  CheckCircle,
  Clock,
  FileText,
} from "lucide-react";
import { Header } from "@/components/layout/header";
import { AntibioticResultRow } from "@/components/features/results/antibiotic-result-row";
import { DetectedGeneCard } from "@/components/features/results/detected-gene-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { predictionService } from "@/services/predictions";
import { formatDateTime, downloadPredictionCSV, downloadPredictionTextReport } from "@/lib/utils";
import type { Prediction } from "@/types";

interface ResultDetailPageProps {
  params: Promise<{ id: string }>;
}

export default function ResultDetailPage({ params }: ResultDetailPageProps) {
  const { id } = use(params);
  const router = useRouter();
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      const data = await predictionService.getById(id);
      if (!data) {
        router.push("/results");
        return;
      }
      setPrediction(data);
      setIsLoading(false);
    };
    loadData();
  }, [id, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!prediction) {
    return null;
  }

  const getRiskConfig = () => {
    switch (prediction.overallRisk) {
      case "high":
        return {
          icon: AlertTriangle,
          color: "text-destructive",
          bgColor: "bg-destructive/10",
          label: "High Risk",
        };
      case "moderate":
        return {
          icon: Clock,
          color: "text-intermediate",
          bgColor: "bg-intermediate/10",
          label: "Moderate Risk",
        };
      case "low":
        return {
          icon: CheckCircle,
          color: "text-susceptible",
          bgColor: "bg-susceptible/10",
          label: "Low Risk",
        };
    }
  };

  const riskConfig = getRiskConfig();
  const RiskIcon = riskConfig.icon;

  return (
    <div className="min-h-screen">
      <Header
        title={`Sample ${prediction.sampleId}`}
        subtitle={prediction.organism}
      />

      <div className="p-6 space-y-6">
        {/* Back & Actions */}
        <div className="flex items-center justify-between">
          <Link href="/results">
            <Button variant="ghost" className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to Results
            </Button>
          </Link>
          <div className="flex gap-2">
            <Button className="gap-2" onClick={() => downloadPredictionTextReport(prediction)}>
              <Download className="h-4 w-4" />
              Export Report
            </Button>
            <Button variant="outline" className="gap-2" onClick={() => downloadPredictionCSV(prediction)}>
              <Download className="h-4 w-4" />
              Export CSV
            </Button>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Overall Risk */}
          <Card className={riskConfig.bgColor}>
            <CardContent className="p-6 text-center">
              <RiskIcon className={`h-8 w-8 mx-auto mb-2 ${riskConfig.color}`} />
              <h3 className={`text-lg font-bold ${riskConfig.color}`}>
                {riskConfig.label}
              </h3>
              <p className="text-sm text-muted">Overall Assessment</p>
            </CardContent>
          </Card>

          {/* Resistant */}
          <Card className="bg-destructive/10">
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-destructive mb-1">
                {prediction.summary?.resistant || 0}
              </div>
              <p className="text-sm text-muted">Resistant</p>
            </CardContent>
          </Card>

          {/* Intermediate */}
          <Card className="bg-intermediate/10">
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-intermediate mb-1">
                {prediction.summary?.intermediate || 0}
              </div>
              <p className="text-sm text-muted">Intermediate</p>
            </CardContent>
          </Card>

          {/* Susceptible */}
          <Card className="bg-susceptible/10">
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-susceptible mb-1">
                {prediction.summary?.susceptible || 0}
              </div>
              <p className="text-sm text-muted">Susceptible</p>
            </CardContent>
          </Card>
        </div>

        {/* Sample Info */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              Sample Information
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-sm text-muted mb-1">Sample ID</p>
                <p className="font-medium">{prediction.sampleId}</p>
              </div>
              <div>
                <p className="text-sm text-muted mb-1">Organism</p>
                <p className="font-medium italic">{prediction.organism}</p>
              </div>
              <div>
                <p className="text-sm text-muted mb-1">Uploaded</p>
                <p className="font-medium">{formatDateTime(prediction.createdAt)}</p>
              </div>
              <div>
                <p className="text-sm text-muted mb-1">Analyzed</p>
                <p className="font-medium">
                  {prediction.completedAt
                    ? formatDateTime(prediction.completedAt)
                    : "Pending"}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted mb-1">File</p>
                <p className="font-medium">{prediction.fileName}</p>
              </div>
              <div>
                <p className="text-sm text-muted mb-1">Uploaded By</p>
                <p className="font-medium">{prediction.uploadedBy}</p>
              </div>
              <div>
                <p className="text-sm text-muted mb-1">Status</p>
                <Badge
                  variant={
                    prediction.status === "completed" ? "success" : "secondary"
                  }
                >
                  {prediction.status}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Detailed Results Tabs */}
        <Tabs defaultValue="antibiotics" className="space-y-4">
          <TabsList>
            <TabsTrigger value="antibiotics">
              Antibiotic Results ({prediction.results?.length || 0})
            </TabsTrigger>
            <TabsTrigger value="genes">
              Detected Genes ({prediction.detectedGenes?.length || 0})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="antibiotics">
            <Card>
              <CardContent className="p-0">
                {prediction.results && prediction.results.length > 0 ? (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Antibiotic</TableHead>
                        <TableHead>Class</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Confidence</TableHead>
                        <TableHead>MIC</TableHead>
                        <TableHead>Breakpoint</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {prediction.results.map((result, index) => (
                        <AntibioticResultRow key={index} result={result} />
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="text-center py-12 text-muted">
                    No antibiotic results available
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="genes">
            {prediction.detectedGenes && prediction.detectedGenes.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {prediction.detectedGenes.map((gene, index) => (
                  <DetectedGeneCard key={index} gene={gene} />
                ))}
              </div>
            ) : (
              <Card>
                <CardContent className="text-center py-12 text-muted">
                  <Dna className="h-12 w-12 mx-auto mb-4 opacity-20" />
                  <p>No resistance genes detected</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
