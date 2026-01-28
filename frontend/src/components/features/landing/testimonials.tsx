import { Quote } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

const testimonials = [
  {
    quote:
      "DeepAMR has transformed how we approach TB treatment in our hospital. We can now identify drug-resistant cases much faster and start appropriate therapy immediately.",
    author: "Dr. Rahim Rahman",
    role: "Pulmonologist",
    organization: "Dhaka Medical College Hospital",
    avatar: null,
  },
  {
    quote:
      "The accuracy of resistance predictions has been remarkable. It's helping us combat the growing MDR-TB problem in Bangladesh more effectively.",
    author: "Dr. Fatima Chowdhury",
    role: "Microbiologist",
    organization: "icddr,b",
    avatar: null,
  },
  {
    quote:
      "As a clinical lab, we've integrated DeepAMR into our workflow. The turnaround time improvement has been significant - from days to hours.",
    author: "Dr. Kamal Islam",
    role: "Laboratory Director",
    organization: "Popular Diagnostic Centre",
    avatar: null,
  },
];

export function Testimonials() {
  return (
    <section className="py-24 bg-background-secondary">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Trusted by <span className="gradient-text">Healthcare Leaders</span>
          </h2>
          <p className="text-lg text-muted max-w-2xl mx-auto">
            See what clinicians and researchers are saying about DeepAMR.
          </p>
        </div>

        {/* Testimonials Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <Card
              key={testimonial.author}
              className="relative animate-slide-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <CardContent className="pt-8">
                {/* Quote Icon */}
                <div className="absolute -top-4 left-6">
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                    <Quote className="h-4 w-4 text-primary-foreground" />
                  </div>
                </div>

                {/* Quote Text */}
                <p className="text-muted mb-6 italic">
                  &ldquo;{testimonial.quote}&rdquo;
                </p>

                {/* Author */}
                <div className="flex items-center gap-3">
                  <Avatar>
                    <AvatarImage src={testimonial.avatar || undefined} />
                    <AvatarFallback className="bg-primary/10 text-primary">
                      {testimonial.author
                        .split(" ")
                        .map((n) => n[0])
                        .join("")}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="text-sm font-medium text-foreground">
                      {testimonial.author}
                    </p>
                    <p className="text-xs text-muted">
                      {testimonial.role}, {testimonial.organization}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
