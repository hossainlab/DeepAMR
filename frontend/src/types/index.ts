// User types
export interface User {
  id: string;
  email: string;
  name: string;
  role: "admin" | "user" | "lab_technician";
  organization?: string;
  avatar?: string;
  createdAt: string;
  lastLogin?: string;
}

// Auth types
export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

// Organism types
export type Organism =
  | "Mycobacterium tuberculosis"
  | "Escherichia coli"
  | "Staphylococcus aureus"
  | "Klebsiella pneumoniae"
  | "Pseudomonas aeruginosa"
  | "Acinetobacter baumannii"
  | "Salmonella typhi"
  | "Vibrio cholerae"
  | "Neisseria gonorrhoeae"
  | "Streptococcus pneumoniae";

// Resistance status
export type ResistanceStatus = "R" | "I" | "S";

// Antibiotic result
export interface AntibioticResult {
  antibiotic: string;
  class: string;
  status: ResistanceStatus;
  confidence: number;
  mic?: string;
  breakpoint?: string;
}

// Detected gene
export interface DetectedGene {
  gene: string;
  description: string;
  mechanism: string;
  coverage: number;
  identity: number;
  antibiotics: string[];
}

// Prediction
export interface Prediction {
  id: string;
  sampleId: string;
  organism: Organism;
  status: "pending" | "processing" | "completed" | "failed";
  createdAt: string;
  completedAt?: string;
  uploadedBy: string;
  fileName: string;
  fileSize: number;
  overallRisk: "high" | "moderate" | "low";
  results?: AntibioticResult[];
  detectedGenes?: DetectedGene[];
  summary?: {
    resistant: number;
    intermediate: number;
    susceptible: number;
  };
}

// Upload types
export interface UploadFile {
  file: File;
  id: string;
  name: string;
  size: number;
  type: string;
  status: "pending" | "uploading" | "validating" | "processing" | "completed" | "error";
  progress: number;
  error?: string;
}

export interface UploadProgress {
  stage: "uploading" | "validating" | "analyzing" | "predicting" | "completed";
  progress: number;
  message: string;
}

// Report types
export interface Report {
  id: string;
  predictionId: string;
  sampleId: string;
  organism: Organism;
  generatedAt: string;
  downloadUrl: string;
  fileSize: number;
}

// Stats types
export interface DashboardStats {
  totalPredictions: number;
  resistantCount: number;
  susceptibleCount: number;
  pendingCount: number;
  weeklyChange: {
    predictions: number;
    resistant: number;
  };
}

// Filter types
export interface PredictionFilters {
  dateFrom?: string;
  dateTo?: string;
  organism?: Organism;
  status?: Prediction["status"];
  risk?: Prediction["overallRisk"];
  search?: string;
}

// Admin types
export interface AdminStats {
  totalUsers: number;
  activeUsers: number;
  totalPredictions: number;
  predictionsToday: number;
  storageUsed: number;
  storageLimit: number;
}

export interface UserActivity {
  userId: string;
  userName: string;
  action: string;
  timestamp: string;
  details?: string;
}

// Chart data types
export interface ChartDataPoint {
  name: string;
  value: number;
  color?: string;
}

export interface TimeSeriesDataPoint {
  date: string;
  resistant: number;
  susceptible: number;
  intermediate: number;
}
