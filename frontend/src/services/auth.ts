import { mockUsers, currentUser } from "@/data/mock-users";
import type { User } from "@/types";
import { delay } from "@/lib/utils";

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

// Mock auth service
export const authService = {
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    await delay(1000); // Simulate network delay

    const user = mockUsers.find(u => u.email === credentials.email);

    if (!user) {
      return { success: false, error: "Invalid email or password" };
    }

    // In a real app, we'd validate the password
    if (credentials.password.length < 6) {
      return { success: false, error: "Invalid email or password" };
    }

    // Store in localStorage for persistence
    if (typeof window !== "undefined") {
      localStorage.setItem("user", JSON.stringify(user));
    }

    return { success: true, user };
  },

  async register(data: RegisterData): Promise<AuthResponse> {
    await delay(1500); // Simulate network delay

    // Check if email already exists
    const existingUser = mockUsers.find(u => u.email === data.email);
    if (existingUser) {
      return { success: false, error: "Email already registered" };
    }

    // Create new user
    const newUser: User = {
      id: `user-${Date.now()}`,
      email: data.email,
      name: data.name,
      role: "user",
      organization: data.organization,
      createdAt: new Date().toISOString(),
      lastLogin: new Date().toISOString(),
    };

    // Store in localStorage
    if (typeof window !== "undefined") {
      localStorage.setItem("user", JSON.stringify(newUser));
    }

    return { success: true, user: newUser };
  },

  async logout(): Promise<void> {
    await delay(500);
    if (typeof window !== "undefined") {
      localStorage.removeItem("user");
    }
  },

  async getCurrentUser(): Promise<User | null> {
    await delay(300);

    if (typeof window === "undefined") {
      return null;
    }

    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      return JSON.parse(storedUser);
    }

    return null;
  },

  async getDemoUser(): Promise<User> {
    return currentUser;
  },
};
