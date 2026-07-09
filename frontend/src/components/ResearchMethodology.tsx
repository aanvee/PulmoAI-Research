import React from "react";
import { TrendingUp, ShieldCheck, Cpu, Award } from "lucide-react";

export default function ResearchMethodology() {
  return (
    <section id="methodology" className="max-w-7xl mx-auto px-6 md:px-12 py-12 space-y-12 scroll-mt-20">
      <div className="text-center space-y-3 max-w-3xl mx-auto">
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary-fixed text-on-primary-fixed-variant text-xs font-bold uppercase tracking-wider">
          <Award className="w-3.5 h-3.5" /> Peer Reviewed Science
        </span>
        <h2 className="font-heading text-3xl sm:text-4xl font-bold text-on-surface">Research Methodology</h2>
        <p className="font-sans text-base sm:text-lg text-secondary">
          Our dual-stream architecture leverages deep vision features and dense neural streams to fuse radiographs with raw patient diagnostics.
        </p>
      </div>

      {/* Bento Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left: Dual-Stream Diagram Card */}
        <div className="lg:col-span-2 glass-card rounded-[32px] p-8 flex flex-col justify-between border border-primary/15 relative overflow-hidden bg-white">
          <div className="absolute -top-12 -right-12 w-48 h-48 bg-primary/5 rounded-full blur-3xl -z-10"></div>
          
          <div>
            <span className="bg-primary-fixed text-on-primary-fixed-variant px-3 py-1 rounded-full text-xs font-bold mb-4 inline-block tracking-wide">
              SYSTEM ARCHITECTURE
            </span>
            <h3 className="font-heading text-xl sm:text-2xl font-bold text-on-surface">DenseNet121 + MLP Attention Fusion</h3>
            <p className="text-sm text-secondary mt-1 max-w-lg">
              Synchronous processing of high-dimensional pixel states and low-dimensional clinical vitals inside a unified cross-modal weight matrix.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-8 relative">
            {/* Vision Stream */}
            <div className="bg-surface p-5 rounded-2xl border border-outline-variant space-y-2 flex flex-col justify-between hover:border-primary transition-colors">
              <div className="flex items-center gap-2 font-bold text-primary">
                <Cpu className="w-4 h-4" />
                <span>Vision Stream</span>
              </div>
              <p className="text-xs text-on-surface-variant leading-relaxed">
                DenseNet-121 pre-trained on 224,316 CheXpert chest radiographs with a custom global average pooling (GAP) layer for robust latent vector representation.
              </p>
            </div>

            {/* Clinical Stream */}
            <div className="bg-surface p-5 rounded-2xl border border-outline-variant space-y-2 flex flex-col justify-between hover:border-primary transition-colors">
              <div className="flex items-center gap-2 font-bold text-secondary">
                <ShieldCheck className="w-4 h-4" />
                <span>Clinical Stream</span>
              </div>
              <p className="text-xs text-on-surface-variant leading-relaxed">
                Multi-layer Perceptron (MLP) mapping 12 channels of patient clinical vitals (Temperature, SpO2, Heart Rate, Respiration Rate) and discrete symptom vectors.
              </p>
            </div>

            {/* Fusion layer block */}
            <div className="col-span-1 sm:col-span-2 flex items-center justify-center p-6 border-2 border-primary-fixed bg-primary-fixed/30 rounded-2xl shadow-sm hover:bg-primary-fixed/40 transition-colors">
              <div className="text-center font-heading text-sm md:text-base text-primary font-extrabold tracking-widest uppercase">
                CROSS-MODAL ATTENTION FUSION LAYER
              </div>
            </div>
          </div>
        </div>

        {/* Right: Metrics Panel */}
        <div className="grid grid-cols-1 gap-4">
          {/* Accuracy Card */}
          <div className="bg-primary text-on-primary p-6 rounded-[24px] shadow-lg flex flex-col justify-between hover:-translate-y-1 transition-all duration-300">
            <div className="text-xs font-bold opacity-80 uppercase tracking-widest font-sans">Validation Accuracy</div>
            <div className="text-4xl sm:text-5xl font-extrabold font-heading tracking-tight mt-2">94.2%</div>
            <div className="flex items-center gap-1 text-xs font-bold text-primary-fixed mt-4">
              <TrendingUp className="w-4 h-4" />
              <span>+1.2% vs SOTA baseline</span>
            </div>
          </div>

          {/* F1 Score Card */}
          <div className="bg-surface-container-highest p-6 rounded-[24px] border border-outline-variant flex flex-col justify-between hover:-translate-y-1 transition-all duration-300">
            <div className="text-xs font-bold text-on-surface-variant uppercase tracking-widest font-sans">F1 Score</div>
            <div className="text-4xl font-extrabold text-on-surface font-heading tracking-tight mt-2">0.91</div>
            <div className="text-secondary text-xs font-semibold mt-4">Balanced clinical precision/recall ratio</div>
          </div>

          {/* ROC-AUC Card */}
          <div className="bg-surface-container-highest p-6 rounded-[24px] border border-outline-variant flex flex-col justify-between hover:-translate-y-1 transition-all duration-300">
            <div className="text-xs font-bold text-on-surface-variant uppercase tracking-widest font-sans">ROC-AUC</div>
            <div className="text-4xl font-extrabold text-on-surface font-heading tracking-tight mt-2">0.96</div>
            <div className="text-secondary text-xs font-semibold mt-4">Excellent diagnostic discrimination threshold</div>
          </div>
        </div>
      </div>
    </section>
  );
}
