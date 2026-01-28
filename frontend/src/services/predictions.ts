import { mockPredictions, antibioticsByOrganism, genesByOrganism, organisms } from "@/data/mock-predictions";
import type { Prediction, PredictionFilters, Organism, UploadProgress } from "@/types";
import { delay, generateId } from "@/lib/utils";

export const predictionService = {
  async getAll(filters?: PredictionFilters): Promise<Prediction[]> {
    await delay(500);

    let results = [...mockPredictions];

    if (filters) {
      if (filters.organism) {
        results = results.filter(p => p.organism === filters.organism);
      }
      if (filters.status) {
        results = results.filter(p => p.status === filters.status);
      }
      if (filters.risk) {
        results = results.filter(p => p.overallRisk === filters.risk);
      }
      if (filters.search) {
        const searchLower = filters.search.toLowerCase();
        results = results.filter(
          p =>
            p.sampleId.toLowerCase().includes(searchLower) ||
            p.organism.toLowerCase().includes(searchLower) ||
            p.fileName.toLowerCase().includes(searchLower)
        );
      }
      if (filters.dateFrom) {
        results = results.filter(p => new Date(p.createdAt) >= new Date(filters.dateFrom!));
      }
      if (filters.dateTo) {
        results = results.filter(p => new Date(p.createdAt) <= new Date(filters.dateTo!));
      }
    }

    return results.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
  },

  async getById(id: string): Promise<Prediction | null> {
    await delay(300);
    return mockPredictions.find(p => p.id === id) || null;
  },

  async getRecent(limit: number = 5): Promise<Prediction[]> {
    await delay(300);
    return mockPredictions
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
      .slice(0, limit);
  },

  async upload(
    file: File,
    organism: Organism,
    onProgress: (progress: UploadProgress) => void
  ): Promise<Prediction> {
    // Stage 1: Uploading
    onProgress({ stage: "uploading", progress: 0, message: "Uploading file..." });
    await delay(500);
    onProgress({ stage: "uploading", progress: 30, message: "Uploading file..." });
    await delay(500);
    onProgress({ stage: "uploading", progress: 60, message: "Uploading file..." });
    await delay(500);
    onProgress({ stage: "uploading", progress: 100, message: "Upload complete" });

    // Stage 2: Validating
    await delay(300);
    onProgress({ stage: "validating", progress: 0, message: "Validating sequence format..." });
    await delay(800);
    onProgress({ stage: "validating", progress: 50, message: "Checking sequence quality..." });
    await delay(800);
    onProgress({ stage: "validating", progress: 100, message: "Validation complete" });

    // Stage 3: Analyzing
    await delay(300);
    onProgress({ stage: "analyzing", progress: 0, message: "Aligning sequences..." });
    await delay(1000);
    onProgress({ stage: "analyzing", progress: 25, message: "Identifying resistance genes..." });
    await delay(1000);
    onProgress({ stage: "analyzing", progress: 50, message: "Mapping mutations..." });
    await delay(1000);
    onProgress({ stage: "analyzing", progress: 75, message: "Analyzing gene coverage..." });
    await delay(1000);
    onProgress({ stage: "analyzing", progress: 100, message: "Analysis complete" });

    // Stage 4: Predicting
    await delay(300);
    onProgress({ stage: "predicting", progress: 0, message: "Running DeepAMR model..." });
    await delay(1500);
    onProgress({ stage: "predicting", progress: 50, message: "Calculating confidence scores..." });
    await delay(1500);
    onProgress({ stage: "predicting", progress: 100, message: "Prediction complete" });

    // Stage 5: Completed
    await delay(300);
    onProgress({ stage: "completed", progress: 100, message: "Results ready!" });

    // Generate mock prediction result
    const results = antibioticsByOrganism[organism] || antibioticsByOrganism["Escherichia coli"];
    const genes = genesByOrganism[organism] || [];

    const summary = {
      resistant: results.filter(r => r.status === "R").length,
      intermediate: results.filter(r => r.status === "I").length,
      susceptible: results.filter(r => r.status === "S").length,
    };

    const resistanceRatio = summary.resistant / results.length;
    const overallRisk: "high" | "moderate" | "low" =
      resistanceRatio >= 0.4 ? "high" : resistanceRatio >= 0.2 ? "moderate" : "low";

    const prediction: Prediction = {
      id: `pred-${generateId()}`,
      sampleId: `${organism.substring(0, 2).toUpperCase()}-${new Date().getFullYear()}-${generateId().toUpperCase()}`,
      organism,
      status: "completed",
      createdAt: new Date().toISOString(),
      completedAt: new Date().toISOString(),
      uploadedBy: "Current User",
      fileName: file.name,
      fileSize: file.size,
      overallRisk,
      results,
      detectedGenes: genes,
      summary,
    };

    return prediction;
  },

  getOrganisms(): Organism[] {
    return organisms;
  },
};
