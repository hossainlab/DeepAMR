"use client";

import Link from "next/link";
import { ArrowRight, Dna, Shield, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-transparent to-transparent" />
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl animate-pulse delay-1000" />

      {/* Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-32 text-center">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-8 animate-fade-in">
          <Dna className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium text-primary">AI-Powered AMR Detection</span>
        </div>

        {/* Headline */}
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-foreground mb-6 animate-slide-up">
          Predict Antimicrobial{" "}
          <span className="gradient-text">Resistance</span>{" "}
          with AI
        </h1>

        {/* Subheadline */}
        <p className="text-lg sm:text-xl text-muted max-w-3xl mx-auto mb-10 animate-slide-up" style={{ animationDelay: "0.1s" }}>
          DeepAMR uses advanced deep learning to analyze genomic sequences and predict
          antimicrobial resistance patterns. Built for Bangladesh healthcare system,
          helping clinicians make informed treatment decisions.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16 animate-slide-up" style={{ animationDelay: "0.2s" }}>
          <Link href="/register">
            <Button size="xl" className="group">
              Get Started Free
              <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </Button>
          </Link>
          <Link href="#how-it-works">
            <Button variant="outline" size="xl">
              See How It Works
            </Button>
          </Link>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto animate-slide-up" style={{ animationDelay: "0.3s" }}>
          <div className="text-center">
            <div className="text-3xl sm:text-4xl font-bold gradient-text mb-2">95%</div>
            <div className="text-sm text-muted">Prediction Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-3xl sm:text-4xl font-bold gradient-text mb-2">12+</div>
            <div className="text-sm text-muted">Organisms Supported</div>
          </div>
          <div className="text-center">
            <div className="text-3xl sm:text-4xl font-bold gradient-text mb-2">50+</div>
            <div className="text-sm text-muted">Antibiotics Covered</div>
          </div>
          <div className="text-center">
            <div className="text-3xl sm:text-4xl font-bold gradient-text mb-2">{"<"}15min</div>
            <div className="text-sm text-muted">Results Time</div>
          </div>
        </div>

        {/* Trust Indicators */}
        <div className="flex items-center justify-center gap-8 mt-16 animate-fade-in" style={{ animationDelay: "0.5s" }}>
          <div className="flex items-center gap-2 text-muted">
            <Shield className="h-5 w-5 text-primary" />
            <span className="text-sm">HIPAA Compliant</span>
          </div>
          <div className="flex items-center gap-2 text-muted">
            <Zap className="h-5 w-5 text-primary" />
            <span className="text-sm">Real-time Analysis</span>
          </div>
        </div>
      </div>
    </section>
  );
}
