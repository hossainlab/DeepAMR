"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, Dna } from "lucide-react";
import { Header } from "@/components/layout/header";
import { FileDropzone } from "@/components/features/upload/file-dropzone";
import { UploadProgressDisplay } from "@/components/features/upload/upload-progress";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { predictionService } from "@/services/predictions";
import type { Organism, UploadProgress, Prediction } from "@/types";

export default function UploadPage() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedOrganism, setSelectedOrganism] = useState<Organism | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);

  const organisms = predictionService.getOrganisms();

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile || !selectedOrganism) return;

    setIsUploading(true);
    setProgress({ stage: "uploading", progress: 0, message: "Starting upload..." });

    try {
      const prediction = await predictionService.upload(
        selectedFile,
        selectedOrganism,
        setProgress
      );
      setResult(prediction);
    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleViewResults = () => {
    if (result) {
      router.push(`/results/${result.id}`);
    }
  };

  const handleNewUpload = () => {
    setSelectedFile(null);
    setSelectedOrganism(null);
    setProgress(null);
    setResult(null);
  };

  return (
    <div className="min-h-screen">
      <Header
        title="Upload Sample"
        subtitle="Upload your genomic sequence file for AMR prediction"
      />

      <div className="p-6 max-w-3xl mx-auto">
        {result ? (
          // Success State
          <Card className="animate-slide-up">
            <CardHeader className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-susceptible/10 flex items-center justify-center">
                <Dna className="h-8 w-8 text-susceptible" />
              </div>
              <CardTitle className="text-2xl text-susceptible">Analysis Complete!</CardTitle>
              <CardDescription>
                Your sample has been analyzed successfully. View the detailed results below.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 rounded-lg bg-background-secondary space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted">Sample ID</span>
                  <span className="font-medium">{result.sampleId}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Organism</span>
                  <span className="font-medium">{result.organism}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Overall Risk</span>
                  <span
                    className={`font-medium ${
                      result.overallRisk === "high"
                        ? "text-destructive"
                        : result.overallRisk === "moderate"
                        ? "text-intermediate"
                        : "text-susceptible"
                    }`}
                  >
                    {result.overallRisk.charAt(0).toUpperCase() + result.overallRisk.slice(1)}
                  </span>
                </div>
                {result.summary && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-muted">Resistant</span>
                      <span className="font-medium text-destructive">{result.summary.resistant}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted">Susceptible</span>
                      <span className="font-medium text-susceptible">{result.summary.susceptible}</span>
                    </div>
                  </>
                )}
              </div>

              <div className="flex gap-4">
                <Button onClick={handleViewResults} className="flex-1">
                  View Detailed Results
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Button>
                <Button variant="outline" onClick={handleNewUpload}>
                  Upload Another
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : isUploading && progress ? (
          // Progress State
          <Card className="animate-fade-in">
            <CardHeader>
              <CardTitle>Analyzing Sample</CardTitle>
              <CardDescription>
                Please wait while we analyze your genomic sequence...
              </CardDescription>
            </CardHeader>
            <CardContent>
              <UploadProgressDisplay progress={progress} />
            </CardContent>
          </Card>
        ) : (
          // Upload Form
          <Card className="animate-slide-up">
            <CardHeader>
              <CardTitle>Upload Genomic Sequence</CardTitle>
              <CardDescription>
                Upload a FASTA or FASTQ file containing your sample&apos;s genomic sequence.
                We support whole genome sequences, amplicon data, and assembled contigs.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* File Upload */}
              <div className="space-y-2">
                <Label>Sequence File</Label>
                <FileDropzone onFileSelect={handleFileSelect} disabled={isUploading} />
              </div>

              {/* Organism Selection */}
              <div className="space-y-2">
                <Label htmlFor="organism">Target Organism</Label>
                <Select
                  value={selectedOrganism || undefined}
                  onValueChange={(value) => setSelectedOrganism(value as Organism)}
                  disabled={isUploading}
                >
                  <SelectTrigger id="organism">
                    <SelectValue placeholder="Select organism..." />
                  </SelectTrigger>
                  <SelectContent>
                    {organisms.map((organism) => (
                      <SelectItem key={organism} value={organism}>
                        {organism}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted">
                  Select the organism species for accurate resistance prediction
                </p>
              </div>

              {/* Submit Button */}
              <Button
                onClick={handleUpload}
                disabled={!selectedFile || !selectedOrganism || isUploading}
                className="w-full"
                size="lg"
              >
                <Dna className="h-5 w-5 mr-2" />
                Start Analysis
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
