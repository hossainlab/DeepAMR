import { Upload, Cpu, FileCheck, ArrowRight } from "lucide-react";

const steps = [
  {
    icon: Upload,
    number: "01",
    title: "Upload Sample",
    description:
      "Upload your FASTA or FASTQ genomic sequence file. We support whole genome sequences, amplicon data, and metagenomic samples.",
  },
  {
    icon: Cpu,
    number: "02",
    title: "AI Analysis",
    description:
      "Our deep learning model analyzes the sequence, identifying resistance genes, mutations, and predicting susceptibility patterns.",
  },
  {
    icon: FileCheck,
    number: "03",
    title: "Get Results",
    description:
      "Receive detailed results with antibiotic recommendations, confidence scores, and detected resistance mechanisms.",
  },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            How It <span className="gradient-text">Works</span>
          </h2>
          <p className="text-lg text-muted max-w-2xl mx-auto">
            Get from sample to actionable insights in three simple steps.
            No complex setup required.
          </p>
        </div>

        {/* Steps */}
        <div className="relative">
          {/* Connection Line (Desktop) */}
          <div className="hidden lg:block absolute top-1/2 left-1/4 right-1/4 h-0.5 bg-gradient-to-r from-primary/50 via-accent/50 to-primary/50" />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 lg:gap-12">
            {steps.map((step, index) => (
              <div
                key={step.title}
                className="relative text-center animate-slide-up"
                style={{ animationDelay: `${index * 0.15}s` }}
              >
                {/* Step Number & Icon */}
                <div className="relative inline-flex items-center justify-center mb-6">
                  <div className="w-20 h-20 rounded-2xl bg-surface border border-border flex items-center justify-center group-hover:border-primary/50 transition-colors">
                    <step.icon className="h-8 w-8 text-primary" />
                  </div>
                  <span className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-primary text-primary-foreground text-sm font-bold flex items-center justify-center">
                    {step.number}
                  </span>
                </div>

                {/* Content */}
                <h3 className="text-xl font-semibold text-foreground mb-3">
                  {step.title}
                </h3>
                <p className="text-muted max-w-sm mx-auto">
                  {step.description}
                </p>

                {/* Arrow (Mobile) */}
                {index < steps.length - 1 && (
                  <div className="lg:hidden flex justify-center my-6">
                    <ArrowRight className="h-6 w-6 text-primary rotate-90" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
