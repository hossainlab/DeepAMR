import { deepamrApi } from "./api";
import type { User } from "@/types";

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  name: string;
  email: string;
  password: string;
  organization?: string;
}

export interface AuthResponse {
  success: boolean;
  user?: User;
  error?: string;
}

export const authService = {
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    try {
      const result = await deepamrApi.auth.login(credentials);
      if (typeof window !== "undefined") {
        localStorage.setItem("user", JSON.stringify(result.user));
      }
      return { success: true, user: result.user };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Login failed";
      return { success: false, error: message };
    }
  },

  async register(data: RegisterData): Promise<AuthResponse> {
    try {
      const result = await deepamrApi.auth.register(data);
      if (typeof window !== "undefined") {
        localStorage.setItem("user", JSON.stringify(result.user));
      }
      return { success: true, user: result.user };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Registration failed";
      return { success: false, error: message };
    }
  },

  async logout(): Promise<void> {
    await deepamrApi.auth.logout();
    if (typeof window !== "undefined") {
      localStorage.removeItem("user");
    }
  },

  async getCurrentUser(): Promise<User | null> {
    if (typeof window === "undefined") return null;

    // Check for stored token first
    const token = localStorage.getItem("token");
    if (!token) {
      localStorage.removeItem("user");
      return null;
    }

    // Try validating against the API
    try {
      const result = await deepamrApi.auth.me();
      localStorage.setItem("user", JSON.stringify(result.user));
      return result.user;
    } catch {
      // Token expired or invalid — clear storage
      localStorage.removeItem("user");
      localStorage.removeItem("token");
      return null;
    }
  },
};
