import Link from "next/link";
import { ArrowRight, Dna } from "lucide-react";
import { Button } from "@/components/ui/button";

export function CTA() {
  return (
    <section className="py-24 bg-background relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-gradient-to-t from-primary/5 via-transparent to-transparent" />
      <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
      <div className="absolute top-0 right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl" />

      <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        {/* Icon */}
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 mb-8 animate-pulse-glow">
          <Dna className="h-8 w-8 text-primary" />
        </div>

        {/* Content */}
        <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-6">
          Ready to Transform Your{" "}
          <span className="gradient-text">AMR Diagnostics</span>?
        </h2>
        <p className="text-lg text-muted mb-10 max-w-2xl mx-auto">
          Join leading healthcare institutions in Bangladesh using DeepAMR to
          make faster, more accurate treatment decisions. Start your free trial today.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link href="/register">
            <Button size="xl" className="group">
              Start Free Trial
              <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </Button>
          </Link>
          <Link href="/contact">
            <Button variant="outline" size="xl">
              Contact Sales
            </Button>
          </Link>
        </div>

        {/* Trust Note */}
        <p className="text-sm text-muted mt-8">
          No credit card required. 14-day free trial with full access.
        </p>
      </div>
    </section>
  );
}
