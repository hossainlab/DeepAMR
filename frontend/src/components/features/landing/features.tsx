import { Zap, Target, Dna, Lock } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const features = [
  {
    icon: Zap,
    title: "Rapid Results",
    description:
      "Get comprehensive AMR predictions in under 15 minutes. No need to wait days for traditional culture-based methods.",
  },
  {
    icon: Target,
    title: "High Accuracy",
    description:
      "Our deep learning model achieves 95%+ accuracy, trained on thousands of validated resistance profiles from clinical isolates.",
  },
  {
    icon: Dna,
    title: "Gene Detection",
    description:
      "Identify specific resistance genes and mutations. Understand the molecular basis of resistance for targeted treatment.",
  },
  {
    icon: Lock,
    title: "Secure & Private",
    description:
      "Your data is encrypted end-to-end. We follow international data protection standards including HIPAA compliance.",
  },
];

export function Features() {
  return (
    <section id="features" className="py-24 bg-background-secondary">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Why Choose <span className="gradient-text">DeepAMR</span>?
          </h2>
          <p className="text-lg text-muted max-w-2xl mx-auto">
            Cutting-edge technology designed specifically for the challenges of
            antimicrobial resistance in Bangladesh and South Asia.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <Card
              key={feature.title}
              hover
              className="group animate-slide-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <CardTitle className="text-lg">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted">{feature.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
