import type { Prediction, PredictionFilters, Organism, UploadProgress, AntibioticResult } from "@/types";
import { generateId } from "@/lib/utils";
import { deepamrApi, convertApiResponseToPrediction } from "./api";

// Check if file is a genomic sequence file (FASTA/FASTQ)
function isSequenceFile(filename: string): boolean {
  const lower = filename.toLowerCase();
  return ['.fasta', '.fa', '.fna', '.fastq', '.fq', '.fasta.gz', '.fa.gz', '.fna.gz', '.fastq.gz', '.fq.gz'].some(ext => lower.endsWith(ext));
}

// Static organism list (reference data, not mock data)
const organisms: Organism[] = [
  "Mycobacterium tuberculosis",
  "Escherichia coli",
  "Staphylococcus aureus",
  "Klebsiella pneumoniae",
  "Pseudomonas aeruginosa",
  "Acinetobacter baumannii",
  "Salmonella typhi",
  "Vibrio cholerae",
  "Neisseria gonorrhoeae",
  "Streptococcus pneumoniae",
];

export const predictionService = {
  async getAll(filters?: PredictionFilters): Promise<Prediction[]> {
    try {
      return await deepamrApi.predictions.list(filters);
    } catch (error) {
      console.error("Failed to fetch predictions:", error);
      return [];
    }
  },

  async getById(id: string): Promise<Prediction | null> {
    try {
      return await deepamrApi.predictions.getById(id);
    } catch (error) {
      console.error("Failed to fetch prediction:", error);
      return null;
    }
  },

  async getRecent(limit: number = 5): Promise<Prediction[]> {
    try {
      return await deepamrApi.predictions.getRecent(limit);
    } catch (error) {
      console.error("Failed to fetch recent predictions:", error);
      return [];
    }
  },

  async upload(
    file: File,
    organism: Organism,
    onProgress: (progress: UploadProgress) => void
  ): Promise<Prediction> {
    // Stage 1: Uploading
    onProgress({ stage: "uploading", progress: 0, message: "Uploading file..." });
    await new Promise(r => setTimeout(r, 300));
    onProgress({ stage: "uploading", progress: 50, message: "Uploading file..." });
    await new Promise(r => setTimeout(r, 300));
    onProgress({ stage: "uploading", progress: 100, message: "Upload complete" });

    // Stage 2: Validating
    await new Promise(r => setTimeout(r, 200));
    onProgress({ stage: "validating", progress: 0, message: "Validating sequence format..." });
    await new Promise(r => setTimeout(r, 400));
    onProgress({ stage: "validating", progress: 100, message: "Validation complete" });

    // Stage 3: Analyzing
    await new Promise(r => setTimeout(r, 200));
    onProgress({ stage: "analyzing", progress: 0, message: "Preparing genomic data..." });
    await new Promise(r => setTimeout(r, 400));
    onProgress({ stage: "analyzing", progress: 100, message: "Analysis complete" });

    // Stage 4: Predicting - call real API
    await new Promise(r => setTimeout(r, 200));
    onProgress({ stage: "predicting", progress: 0, message: "Running DeepAMR model..." });

    let results: AntibioticResult[];
    let summary: { resistant: number; intermediate: number; susceptible: number };
    let overallRisk: "high" | "moderate" | "low";
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let apiRaw: any = null;

    if (isSequenceFile(file.name)) {
      onProgress({ stage: "predicting", progress: 20, message: "Uploading to DeepAMR API..." });
      apiRaw = await deepamrApi.predictFromFasta(file, 0.5, 'deep_learning', organism);
      onProgress({ stage: "predicting", progress: 80, message: "Processing results..." });

      const converted = convertApiResponseToPrediction(apiRaw, {
        fileName: file.name,
        fileSize: file.size,
        organism,
      });

      results = converted.results.map(r => ({
        antibiotic: r.antibiotic,
        class: r.class,
        status: r.status,
        confidence: r.confidence,
      }));
      summary = converted.summary;
      overallRisk = converted.overallRisk;
    } else if (file.name.endsWith('.json')) {
      const text = await file.text();
      const data = JSON.parse(text);
      const features: number[] = Array.isArray(data) ? data : data.features;

      if (features && features.length >= 100) {
        onProgress({ stage: "predicting", progress: 30, message: "Connecting to DeepAMR API..." });
        apiRaw = await deepamrApi.predict({ features, threshold: 0.5 });
        onProgress({ stage: "predicting", progress: 80, message: "Processing results..." });

        const converted = convertApiResponseToPrediction(apiRaw, {
          fileName: file.name,
          fileSize: file.size,
          organism,
        });

        results = converted.results.map(r => ({
          antibiotic: r.antibiotic,
          class: r.class,
          status: r.status,
          confidence: r.confidence,
        }));
        summary = converted.summary;
        overallRisk = converted.overallRisk;
      } else {
        throw new Error("Insufficient features in JSON file");
      }
    } else {
      throw new Error("Unsupported file format. Please upload a FASTA, FASTQ, or JSON file.");
    }

    onProgress({ stage: "predicting", progress: 100, message: "Prediction complete" });

    // Stage 5: Completed
    await new Promise(r => setTimeout(r, 200));
    onProgress({ stage: "completed", progress: 100, message: "Results ready!" });

    // Use backend-generated IDs if available, otherwise fall back to local
    const predictionId = apiRaw?.prediction_id || `pred-${generateId()}`;
    const sampleId = apiRaw?.sample_id || `${organism.substring(0, 2).toUpperCase()}-${new Date().getFullYear()}-${generateId().toUpperCase()}`;

    const prediction: Prediction = {
      id: predictionId,
      sampleId,
      organism,
      status: "completed",
      createdAt: new Date().toISOString(),
      completedAt: new Date().toISOString(),
      uploadedBy: "Current User",
      fileName: file.name,
      fileSize: file.size,
      overallRisk,
      results,
      detectedGenes: [],
      summary,
    };

    return prediction;
  },

  getOrganisms(): Organism[] {
    return organisms;
  },
};
