import React from "react";
import { ArrowRight, Sparkles } from "lucide-react";

export default function Hero() {
  const handleStartPrediction = () => {
    const el = document.getElementById("prediction");
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <section className="relative min-h-[500px] flex flex-col items-center justify-center text-center px-6 md:px-12 py-16 overflow-hidden bg-gradient-to-b from-surface-container-low/30 via-background to-background mt-16">
      {/* Decorative blurry background circles */}
      <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-primary-fixed/20 rounded-full blur-3xl -z-10 animate-pulse"></div>
      <div className="absolute bottom-1/4 right-1/4 w-72 h-72 bg-secondary-fixed/20 rounded-full blur-3xl -z-10"></div>

      <div className="max-w-4xl space-y-6">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary-fixed text-on-primary-fixed-variant text-xs font-semibold tracking-wide uppercase">
          <Sparkles className="w-3.5 h-3.5" />
          Attention-Enhanced Multimodal AI
        </div>

        <h1 className="font-heading text-4xl sm:text-5xl md:text-[52px] leading-tight text-on-surface tracking-tight font-extrabold max-w-4xl mx-auto">
          Attention-Enhanced Multimodal Lung Disease Detection using Chest X-rays and Clinical Information
        </h1>

        <p className="font-sans text-base sm:text-lg md:text-xl text-secondary max-w-2xl mx-auto leading-relaxed">
          A state-of-the-art diagnostic pipeline combining deep vision features with patient clinical data for high-fidelity disease prediction and explainable localization.
        </p>

        <div className="pt-6">
          <button
            onClick={handleStartPrediction}
            className="inline-flex items-center gap-2 bg-primary text-on-primary px-8 py-4 rounded-full font-bold text-base hover:bg-primary-container shadow-lg hover:shadow-xl hover:translate-y-[-1px] transition-all active:scale-95 cursor-pointer group"
          >
            Start Prediction
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </div>
    </section>
  );
}
