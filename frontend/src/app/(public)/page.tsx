import { Navbar } from "@/components/layout/navbar";
import { Footer } from "@/components/layout/footer";
import { Hero } from "@/components/features/landing/hero";
import { Features } from "@/components/features/landing/features";
import { HowItWorks } from "@/components/features/landing/how-it-works";
import { Testimonials } from "@/components/features/landing/testimonials";
import { CTA } from "@/components/features/landing/cta";

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      <Navbar />
      <main>
        <Hero />
        <Features />
        <HowItWorks />
        <Testimonials />
        <CTA />
      </main>
      <Footer />
    </div>
  );
}
