/**
 * DeepAMR API Service
 * Connects to the FastAPI backend for real AMR predictions
 */

import type { User, Prediction, DashboardStats, AdminStats, UserActivity, PredictionFilters } from "@/types";

// API Configuration
// Use the Next.js rewrite proxy (/api -> localhost:8000) to avoid CORS issues.
// In production, set NEXT_PUBLIC_API_URL to the real backend origin.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api';

// Types for API responses
export interface DrugPrediction {
  drug_class: string;
  resistant: boolean;
  probability: number;
  confidence: 'high' | 'medium' | 'low';
}

export interface PredictionSummary {
  resistant_count: number;
  susceptible_count: number;
  risk_level: 'MINIMAL' | 'LOW' | 'MODERATE' | 'HIGH';
  risk_description: string;
}

export interface DetailedPredictionResponse {
  summary: PredictionSummary;
  drug_predictions: DrugPrediction[];
  recommendations: string[];
  timestamp: string;
}

export interface PredictionRequest {
  features: number[];
  threshold?: number;
  model_type?: 'deep_learning' | 'sklearn';
}

export interface SequenceInfo {
  filename: string;
  format: string;
  sequences_processed: number;
  sequence_headers: string[];
  n_features_extracted: number;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  models_loaded: string[];
}

export interface DrugClassInfo {
  drug_classes: string[];
  count: number;
  descriptions: Record<string, string>;
}

// API Error handling
export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, error.detail || 'Request failed');
  }
  return response.json();
}

// Token management
function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('token');
}

function setToken(token: string) {
  if (typeof window !== 'undefined') {
    localStorage.setItem('token', token);
  }
}

function clearToken() {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('token');
  }
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  if (token) {
    return { Authorization: `Bearer ${token}` };
  }
  return {};
}

// API Service
export const deepamrApi = {
  // -------------------------------------------------------------------
  // Health / Info
  // -------------------------------------------------------------------
  async checkHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return handleResponse<HealthResponse>(response);
  },

  async getDrugClasses(): Promise<DrugClassInfo> {
    const response = await fetch(`${API_BASE_URL}/drug-classes`);
    return handleResponse<DrugClassInfo>(response);
  },

  async getModelInfo(modelType: string = 'deep_learning') {
    const response = await fetch(`${API_BASE_URL}/models/${modelType}/info`);
    return handleResponse(response);
  },

  // -------------------------------------------------------------------
  // Auth
  // -------------------------------------------------------------------
  auth: {
    async register(data: { email: string; name: string; password: string; organization?: string }): Promise<{ user: User; token: string }> {
      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      const result = await handleResponse<{ user: User; token: string }>(response);
      setToken(result.token);
      return result;
    },

    async login(credentials: { email: string; password: string }): Promise<{ user: User; token: string }> {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials),
      });
      const result = await handleResponse<{ user: User; token: string }>(response);
      setToken(result.token);
      return result;
    },

    async me(): Promise<{ user: User }> {
      const response = await fetch(`${API_BASE_URL}/auth/me`, {
        headers: authHeaders(),
      });
      return handleResponse<{ user: User }>(response);
    },

    async logout(): Promise<void> {
      await fetch(`${API_BASE_URL}/auth/logout`, {
        method: 'POST',
        headers: authHeaders(),
      }).catch(() => {});
      clearToken();
    },
  },

  // -------------------------------------------------------------------
  // Predictions (ML)
  // -------------------------------------------------------------------
  async predict(request: PredictionRequest): Promise<DetailedPredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict/detailed`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authHeaders(),
      },
      body: JSON.stringify({
        features: request.features,
        threshold: request.threshold || 0.5,
        model_type: request.model_type || 'deep_learning',
      }),
    });
    return handleResponse<DetailedPredictionResponse>(response);
  },

  async predictBatch(samples: number[][], threshold?: number, modelType?: string) {
    const response = await fetch(`${API_BASE_URL}/predict/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authHeaders(),
      },
      body: JSON.stringify({
        samples,
        threshold: threshold || 0.5,
        model_type: modelType || 'deep_learning',
      }),
    });
    return handleResponse(response);
  },

  async predictFromFasta(
    file: File,
    threshold: number = 0.5,
    modelType: string = 'deep_learning',
    organism: string = 'Unknown',
  ): Promise<DetailedPredictionResponse & { sequence_info: SequenceInfo }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('threshold', threshold.toString());
    formData.append('model_type', modelType);
    formData.append('organism', organism);

    const response = await fetch(`${API_BASE_URL}/predict/fasta`, {
      method: 'POST',
      headers: authHeaders(),
      body: formData,
    });
    return handleResponse(response);
  },

  // -------------------------------------------------------------------
  // Prediction History
  // -------------------------------------------------------------------
  predictions: {
    async list(filters?: PredictionFilters): Promise<Prediction[]> {
      const params = new URLSearchParams();
      if (filters?.organism) params.set('organism', filters.organism);
      if (filters?.status) params.set('status', filters.status);
      if (filters?.risk) params.set('risk', filters.risk);
      if (filters?.search) params.set('search', filters.search);
      if (filters?.dateFrom) params.set('dateFrom', filters.dateFrom);
      if (filters?.dateTo) params.set('dateTo', filters.dateTo);
      const qs = params.toString();
      const response = await fetch(`${API_BASE_URL}/predictions${qs ? `?${qs}` : ''}`, {
        headers: authHeaders(),
      });
      return handleResponse<Prediction[]>(response);
    },

    async getById(id: string): Promise<Prediction> {
      const response = await fetch(`${API_BASE_URL}/predictions/${id}`, {
        headers: authHeaders(),
      });
      return handleResponse<Prediction>(response);
    },

    async getRecent(limit: number = 5): Promise<Prediction[]> {
      const response = await fetch(`${API_BASE_URL}/predictions/recent?limit=${limit}`, {
        headers: authHeaders(),
      });
      return handleResponse<Prediction[]>(response);
    },
  },

  // -------------------------------------------------------------------
  // Dashboard
  // -------------------------------------------------------------------
  dashboard: {
    async getStats(): Promise<DashboardStats> {
      const response = await fetch(`${API_BASE_URL}/dashboard/stats`, {
        headers: authHeaders(),
      });
      return handleResponse<DashboardStats>(response);
    },

    async getResistanceOverview(): Promise<Array<{ name: string; value: number; color: string }>> {
      const response = await fetch(`${API_BASE_URL}/dashboard/resistance-overview`, {
        headers: authHeaders(),
      });
      return handleResponse(response);
    },

    async getTrends(): Promise<Array<{ date: string; resistant: number; susceptible: number; intermediate: number }>> {
      const response = await fetch(`${API_BASE_URL}/dashboard/trends`, {
        headers: authHeaders(),
      });
      return handleResponse(response);
    },
  },

  // -------------------------------------------------------------------
  // Admin
  // -------------------------------------------------------------------
  admin: {
    async getStats(): Promise<AdminStats> {
      const response = await fetch(`${API_BASE_URL}/admin/stats`, {
        headers: authHeaders(),
      });
      return handleResponse<AdminStats>(response);
    },

    async getUsers(): Promise<User[]> {
      const response = await fetch(`${API_BASE_URL}/admin/users`, {
        headers: authHeaders(),
      });
      return handleResponse<User[]>(response);
    },

    async deleteUser(id: string): Promise<void> {
      const response = await fetch(`${API_BASE_URL}/admin/users/${id}`, {
        method: 'DELETE',
        headers: authHeaders(),
      });
      await handleResponse(response);
    },

    async getActivity(): Promise<UserActivity[]> {
      const response = await fetch(`${API_BASE_URL}/admin/activity`, {
        headers: authHeaders(),
      });
      return handleResponse<UserActivity[]>(response);
    },
  },
};

// Drug class mapping (for display purposes)
export const DRUG_CLASS_DISPLAY_NAMES: Record<string, string> = {
  aminoglycoside: 'Aminoglycosides',
  'beta-lactam': 'Beta-lactams',
  fosfomycin: 'Fosfomycin',
  glycopeptide: 'Glycopeptides',
  macrolide: 'Macrolides',
  phenicol: 'Phenicols',
  quinolone: 'Quinolones',
  rifampicin: 'Rifampicin',
  sulfonamide: 'Sulfonamides',
  tetracycline: 'Tetracyclines',
  trimethoprim: 'Trimethoprim',
};

export const DRUG_CLASS_EXAMPLES: Record<string, string[]> = {
  aminoglycoside: ['Gentamicin', 'Amikacin', 'Tobramycin', 'Streptomycin'],
  'beta-lactam': ['Ampicillin', 'Amoxicillin', 'Ceftriaxone', 'Meropenem'],
  fosfomycin: ['Fosfomycin'],
  glycopeptide: ['Vancomycin', 'Teicoplanin'],
  macrolide: ['Azithromycin', 'Erythromycin', 'Clarithromycin'],
  phenicol: ['Chloramphenicol'],
  quinolone: ['Ciprofloxacin', 'Levofloxacin', 'Moxifloxacin'],
  rifampicin: ['Rifampicin', 'Rifabutin'],
  sulfonamide: ['Sulfamethoxazole'],
  tetracycline: ['Tetracycline', 'Doxycycline', 'Minocycline'],
  trimethoprim: ['Trimethoprim', 'Co-trimoxazole'],
};

// Convert API response to frontend Prediction format
export function convertApiResponseToPrediction(
  response: DetailedPredictionResponse,
  sampleInfo: { fileName: string; fileSize: number; organism: string }
): {
  results: Array<{
    antibiotic: string;
    class: string;
    status: 'R' | 'I' | 'S';
    confidence: number;
  }>;
  summary: {
    resistant: number;
    intermediate: number;
    susceptible: number;
  };
  overallRisk: 'high' | 'moderate' | 'low';
  recommendations: string[];
} {
  const results = response.drug_predictions.map(drug => ({
    antibiotic: DRUG_CLASS_DISPLAY_NAMES[drug.drug_class] || drug.drug_class,
    class: drug.drug_class,
    status: drug.resistant ? 'R' as const : 'S' as const,
    confidence: drug.probability,
  }));

  const riskMap: Record<string, 'high' | 'moderate' | 'low'> = {
    HIGH: 'high',
    MODERATE: 'moderate',
    LOW: 'low',
    MINIMAL: 'low',
  };

  return {
    results,
    summary: {
      resistant: response.summary.resistant_count,
      intermediate: 0,
      susceptible: response.summary.susceptible_count,
    },
    overallRisk: riskMap[response.summary.risk_level] || 'moderate',
    recommendations: response.recommendations,
  };
}

export default deepamrApi;
