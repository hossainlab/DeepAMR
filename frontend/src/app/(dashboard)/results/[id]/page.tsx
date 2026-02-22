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
  BarChart3,
  Stethoscope,
  Globe,
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
import { deepamrApi } from "@/services/api";
import type { ModelPerformance } from "@/services/api";
import { formatDateTime, downloadPredictionCSV, downloadPredictionTextReport } from "@/lib/utils";
import type { Prediction } from "@/types";

interface ResultDetailPageProps {
  params: Promise<{ id: string }>;
}

export default function ResultDetailPage({ params }: ResultDetailPageProps) {
  const { id } = use(params);
  const router = useRouter();
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [modelPerf, setModelPerf] = useState<ModelPerformance | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      const [data, perf] = await Promise.all([
        predictionService.getById(id),
        deepamrApi.getModelPerformance().catch(() => null),
      ]);
      if (!data) {
        router.push("/results");
        return;
      }
      setPrediction(data);
      setModelPerf(perf);
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
            <a href={deepamrApi.getPredictionReportUrl(prediction.id)} target="_blank" rel="noopener noreferrer">
              <Button className="gap-2">
                <Download className="h-4 w-4" />
                Export PDF Report
              </Button>
            </a>
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

        {/* Model Accuracy Info */}
        {modelPerf && (
          <Card className="border-primary/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <BarChart3 className="h-4 w-4 text-primary" />
                <span className="font-semibold text-sm">Model Accuracy</span>
                <span className="text-xs text-muted ml-auto">v{modelPerf.model_version}</span>
              </div>
              <div className="grid grid-cols-4 gap-3 text-center">
                <div>
                  <div className="text-lg font-bold text-primary">{(modelPerf.overall.micro_f1 * 100).toFixed(1)}%</div>
                  <div className="text-xs text-muted">F1 Score</div>
                </div>
                <div>
                  <div className="text-lg font-bold text-primary">{(modelPerf.overall.auc * 100).toFixed(1)}%</div>
                  <div className="text-xs text-muted">AUC-ROC</div>
                </div>
                <div>
                  <div className="text-lg font-bold text-primary">{(modelPerf.overall.macro_f1 * 100).toFixed(1)}%</div>
                  <div className="text-xs text-muted">Macro F1</div>
                </div>
                <div>
                  <div className="text-lg font-bold text-primary">{(modelPerf.overall.hamming_loss * 100).toFixed(1)}%</div>
                  <div className="text-xs text-muted">Error Rate</div>
                </div>
              </div>
              <p className="text-xs text-muted mt-2">
                These results are AI-generated predictions. Always confirm with laboratory susceptibility testing before clinical use.
              </p>
            </CardContent>
          </Card>
        )}

        {/* Detailed Results Tabs */}
        <Tabs defaultValue="antibiotics" className="space-y-4">
          <TabsList>
            <TabsTrigger value="antibiotics">
              Antibiotic Results ({prediction.results?.length || 0})
            </TabsTrigger>
            <TabsTrigger value="recommendations">
              Clinical Recommendations
            </TabsTrigger>
            <TabsTrigger value="genes">
              Detected Genes ({prediction.detectedGenes?.length || 0})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="antibiotics">
            <Card>
              <CardContent className="p-0">
                {prediction.results && prediction.results.length > 0 ? (
                  <>
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
                    {/* Confidence bars */}
                    <div className="p-4 border-t">
                      <h4 className="text-sm font-semibold mb-3">Resistance Confidence</h4>
                      <div className="space-y-2">
                        {prediction.results.map((result, index) => {
                          const conf = result.confidence * 100;
                          const isResistant = result.status === "R";
                          const barColor = isResistant
                            ? conf >= 80 ? "bg-red-500" : conf >= 60 ? "bg-yellow-500" : "bg-orange-400"
                            : conf <= 20 ? "bg-green-500" : conf <= 40 ? "bg-green-400" : "bg-yellow-400";
                          return (
                            <div key={index} className="flex items-center gap-3">
                              <span className="text-xs w-28 truncate">{result.antibiotic}</span>
                              <div className="flex-1 bg-muted/30 rounded-full h-2.5">
                                <div
                                  className={`h-2.5 rounded-full ${barColor}`}
                                  style={{ width: `${conf}%` }}
                                />
                              </div>
                              <span className="text-xs w-12 text-right">{conf.toFixed(0)}%</span>
                              <Badge variant={isResistant ? "destructive" : "success"} className="text-xs w-6 justify-center">
                                {result.status}
                              </Badge>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-12 text-muted">
                    No antibiotic results available
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="recommendations">
            <div className="space-y-4">
              {/* Clinical Recommendations */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Stethoscope className="h-5 w-5 text-primary" />
                    Clinical Recommendations
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {prediction.results && prediction.results.length > 0 ? (
                    <ul className="space-y-3">
                      {(() => {
                        const resistant = prediction.results.filter(r => r.status === "R");
                        const susceptible = prediction.results.filter(r => r.status === "S");
                        const recs: string[] = [];

                        if (resistant.length >= 5) {
                          recs.push("URGENT: Multi-drug resistance detected. Recommend immediate infectious disease consultation.");
                          recs.push("Consider combination therapy or reserve antibiotics.");
                        }
                        const resistantClasses = new Set(resistant.map(r => r.class));
                        if (resistantClasses.has("beta-lactam")) {
                          recs.push("Beta-lactam resistance detected. Consider carbapenem or non-beta-lactam alternatives.");
                        }
                        if (resistantClasses.has("quinolone")) {
                          recs.push("Fluoroquinolone resistance detected. Avoid ciprofloxacin/levofloxacin.");
                        }
                        const highConfSusc = susceptible.filter(r => r.confidence < 0.2);
                        if (highConfSusc.length > 0) {
                          recs.push(`High-confidence susceptibility predicted for: ${highConfSusc.slice(0, 3).map(r => r.antibiotic).join(", ")}`);
                        }
                        if (recs.length === 0) {
                          recs.push("Standard antibiotic therapy likely effective. Monitor treatment response.");
                        }
                        return recs.map((rec, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <span className="text-primary mt-0.5">•</span>
                            <span className="text-sm">{rec}</span>
                          </li>
                        ));
                      })()}
                    </ul>
                  ) : (
                    <p className="text-muted text-sm">No results available for recommendations.</p>
                  )}
                </CardContent>
              </Card>

              {/* Bangladesh Clinical Context */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Globe className="h-5 w-5 text-green-600" />
                    Bangladesh Clinical Context
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {prediction.results && prediction.results.length > 0 ? (
                    <ul className="space-y-3">
                      {(() => {
                        const resistantClasses = new Set(
                          prediction.results.filter(r => r.status === "R").map(r => r.class)
                        );
                        const recs: string[] = [];

                        if (resistantClasses.has("quinolone")) {
                          recs.push("Quinolone resistance is ~70% locally. DGHS recommends avoiding ciprofloxacin for empirical therapy. Use ceftriaxone for enteric fever.");
                        }
                        if (resistantClasses.has("beta-lactam")) {
                          recs.push("ESBL prevalence is very high (>60%). Consider carbapenems for serious infections. For UTI, fosfomycin or nitrofurantoin remain effective.");
                        }
                        if (resistantClasses.has("rifampicin")) {
                          recs.push("Possible MDR-TB. Refer to National TB Programme (NTP). Contact NIDCH, Mohakhali, Dhaka.");
                        }
                        if (resistantClasses.size >= 5) {
                          recs.push("Extensive drug resistance detected. Recommend referral to icddr,b or IEDCR for susceptibility testing.");
                        }
                        if (recs.length === 0) {
                          recs.push("Predicted susceptibility aligns with local empirical options. Follow DGHS standard treatment guidelines.");
                        }
                        return recs.map((rec, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <span className="text-green-600 mt-0.5">•</span>
                            <span className="text-sm">{rec}</span>
                          </li>
                        ));
                      })()}
                    </ul>
                  ) : (
                    <p className="text-muted text-sm">No results available.</p>
                  )}
                </CardContent>
              </Card>
            </div>
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
