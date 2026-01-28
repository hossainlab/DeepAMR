"use client";

import { create } from "zustand";
import { authService, type LoginCredentials, type RegisterData } from "@/services/auth";
import type { User, AuthState } from "@/types";

interface AuthStore extends AuthState {
  login: (credentials: LoginCredentials) => Promise<{ success: boolean; error?: string }>;
  register: (data: RegisterData) => Promise<{ success: boolean; error?: string }>;
  logout: () => Promise<void>;
  checkAuth: () => Promise<void>;
  setUser: (user: User | null) => void;
}

export const useAuth = create<AuthStore>((set) => ({
  user: null,
  isAuthenticated: false,
  isLoading: true,

  setUser: (user) => {
    set({ user, isAuthenticated: !!user, isLoading: false });
  },

  login: async (credentials) => {
    set({ isLoading: true });
    const response = await authService.login(credentials);

    if (response.success && response.user) {
      set({ user: response.user, isAuthenticated: true, isLoading: false });
      return { success: true };
    }

    set({ isLoading: false });
    return { success: false, error: response.error };
  },

  register: async (data) => {
    set({ isLoading: true });
    const response = await authService.register(data);

    if (response.success && response.user) {
      set({ user: response.user, isAuthenticated: true, isLoading: false });
      return { success: true };
    }

    set({ isLoading: false });
    return { success: false, error: response.error };
  },

  logout: async () => {
    set({ isLoading: true });
    await authService.logout();
    set({ user: null, isAuthenticated: false, isLoading: false });
  },

  checkAuth: async () => {
    set({ isLoading: true });
    const user = await authService.getCurrentUser();
    set({ user, isAuthenticated: !!user, isLoading: false });
  },
}));
