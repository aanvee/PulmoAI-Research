import React from "react";
import { AlertTriangle, Info, Layers, BarChart2, CheckCircle } from "lucide-react";
import { AnalysisResult } from "../types";

interface AnalysisResultsProps {
  result: AnalysisResult | null;
  xrayUrl: string;
  heatmapUrl: string;
  isLoading: boolean;
}

export default function AnalysisResults({ result, xrayUrl, heatmapUrl, isLoading }: AnalysisResultsProps) {
  if (isLoading) {
    return (
      <section id="results-section" className="max-w-7xl mx-auto px-6 md:px-12 py-12 bg-surface-container-low rounded-[40px] my-12 animate-pulse flex flex-col items-center justify-center min-h-[350px]">
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="w-12 h-12 rounded-full border-4 border-primary border-t-transparent animate-spin"></div>
          <p className="font-heading text-lg font-bold text-on-surface">Fusing Vision & Clinical Vectors...</p>
          <p className="text-sm text-secondary max-w-sm">
            Leveraging server-side model processing to run cross-modal attention maps and generate diagnostic reports.
          </p>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <div className="max-w-7xl mx-auto px-6 md:px-12 py-8 text-center text-secondary border border-dashed border-outline-variant rounded-3xl my-6">
        <Info className="w-8 h-8 mx-auto mb-2 opacity-60" />
        <p className="text-sm font-medium">Please select clinical attributes and click 'Analyze Patient' to calculate prediction weights.</p>
      </div>
    );
  }

  const { pneumonia, pleuralEffusion, pneumothorax } = result.probabilities;
  const isHighRisk = pneumonia > 60 || pleuralEffusion > 60 || pneumothorax > 60;

  return (
    <section id="results-section" className="max-w-7xl mx-auto px-6 md:px-12 py-12 bg-surface-container-low rounded-[40px] my-12 scroll-mt-20">
      {/* Header */}
      <div className="mb-8">
        <h2 className="font-heading text-2xl sm:text-3xl font-bold text-on-surface flex items-center gap-2">
          <span>📊</span> Analysis Results
        </h2>
        <p className="font-sans text-sm text-secondary mt-1">Based on Multimodal Attention Fusion & Gemini Clinical Interpreter</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
        {/* Left: Pathology Predictions */}
        <div className="xl:col-span-5 bg-surface-container-lowest p-6 rounded-3xl shadow-soft border border-outline-variant flex flex-col justify-between">
          <div>
            <h4 className="font-heading text-lg font-bold mb-6 text-on-surface flex items-center gap-2">
              <Layers className="w-5 h-5 text-primary" />
              Predicted Pathologies
            </h4>

            <div className="space-y-6">
              {/* Pneumonia */}
              <div className="space-y-2">
                <div className="flex justify-between items-end">
                  <span className="font-sans text-sm font-bold text-on-surface">Pneumonia</span>
                  <span className={`font-mono text-sm font-extrabold ${pneumonia > 50 ? 'text-error' : 'text-primary'}`}>
                    {pneumonia}%
                  </span>
                </div>
                <div className="w-full bg-surface-container rounded-full h-3 overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-1000 ${pneumonia > 50 ? 'bg-error' : 'bg-primary'}`}
                    style={{ width: `${pneumonia}%` }}
                  ></div>
                </div>
              </div>

              {/* Pleural Effusion */}
              <div className="space-y-2">
                <div className="flex justify-between items-end">
                  <span className="font-sans text-sm font-semibold text-on-surface">Pleural Effusion</span>
                  <span className={`font-mono text-sm font-bold ${pleuralEffusion > 50 ? 'text-error' : 'text-secondary'}`}>
                    {pleuralEffusion}%
                  </span>
                </div>
                <div className="w-full bg-surface-container rounded-full h-3 overflow-hidden">
                  <div 
                    className="h-full bg-secondary rounded-full transition-all duration-1000"
                    style={{ width: `${pleuralEffusion}%` }}
                  ></div>
                </div>
              </div>

              {/* Pneumothorax */}
              <div className="space-y-2">
                <div className="flex justify-between items-end">
                  <span className="font-sans text-sm font-semibold text-on-surface">Pneumothorax</span>
                  <span className={`font-mono text-sm font-bold ${pneumothorax > 50 ? 'text-error' : 'text-tertiary-container'}`}>
                    {pneumothorax}%
                  </span>
                </div>
                <div className="w-full bg-surface-container rounded-full h-3 overflow-hidden">
                  <div 
                    className="h-full bg-tertiary-container rounded-full transition-all duration-1000"
                    style={{ width: `${pneumothorax}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* Action recommendation callout */}
          <div className={`mt-8 p-4 rounded-2xl flex items-start gap-3 border ${
            isHighRisk 
              ? 'bg-error-container/40 border-error/20 text-on-error-container' 
              : 'bg-primary-fixed/40 border-primary/20 text-on-primary-fixed-variant'
          }`}>
            {isHighRisk ? (
              <AlertTriangle className="w-5 h-5 text-error shrink-0 mt-0.5" />
            ) : (
              <CheckCircle className="w-5 h-5 text-primary shrink-0 mt-0.5" />
            )}
            <p className="text-sm font-medium leading-relaxed">
              {result.recommendations[0] || "Clinical correlation is advised."}
            </p>
          </div>
        </div>

        {/* Right: Visual Grad-CAM */}
        <div className="xl:col-span-7 bg-surface-container-lowest p-6 rounded-3xl shadow-soft border border-outline-variant flex flex-col justify-between">
          <div className="flex justify-between items-start gap-4 flex-wrap mb-6">
            <div>
              <h4 className="font-heading text-lg font-bold text-on-surface">Interpretability: Grad-CAM Visualization</h4>
              <p className="text-xs text-secondary mt-1">Saliency maps indicating vision stream network focus areas</p>
            </div>
            <span className="bg-secondary-fixed text-on-secondary-fixed-variant px-3 py-1 rounded-full font-sans text-xs font-semibold">
              Global Average Pooling Layer
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Original Input */}
            <div className="relative rounded-2xl overflow-hidden aspect-[4/3] border border-outline-variant shadow-sm group">
              <img 
                referrerPolicy="no-referrer"
                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" 
                alt="Chest X-Ray input" 
                src={xrayUrl} 
              />
              <div className="absolute top-3 left-3 bg-inverse-surface/80 text-inverse-on-surface px-3 py-1 rounded-full text-xs font-semibold backdrop-blur-sm shadow-sm">
                Original Input
              </div>
            </div>

            {/* Attention Heatmap */}
            <div className="relative rounded-2xl overflow-hidden aspect-[4/3] border border-outline-variant shadow-sm group">
              <img 
                referrerPolicy="no-referrer"
                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" 
                alt="Attention mapping heatmap" 
                src={heatmapUrl} 
              />
              <div className="absolute top-3 left-3 bg-inverse-surface/80 text-inverse-on-surface px-3 py-1 rounded-full text-xs font-semibold backdrop-blur-sm shadow-sm">
                Attention Heatmap
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Dynamic AI Clinical Report Summary */}
      <div className="mt-8 bg-surface-container-lowest p-6 rounded-3xl shadow-soft border border-outline-variant">
        <h4 className="font-heading text-lg font-bold mb-3 text-primary flex items-center gap-2">
          <span>🧠</span> AI Interpreter Diagnostic Summary
        </h4>
        <div className="text-sm text-on-surface-variant leading-relaxed font-sans prose max-w-none">
          {result.analysis}
        </div>
      </div>

      {/* SHAP Feature Importance */}
      <div className="mt-8 bg-surface-container-lowest p-6 rounded-3xl shadow-soft border border-outline-variant">
        <h4 className="font-heading text-lg font-bold mb-6 text-on-surface flex items-center gap-2">
          <BarChart2 className="w-5 h-5 text-primary" />
          Multimodal Feature Importance (SHAP)
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-4">
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-bold text-on-surface">
                <span className="text-secondary">X-Ray (DenseNet-121 Latent)</span>
                <span className="text-primary">{result.shap.xray}</span>
              </div>
              <div className="w-full h-2 bg-surface-container rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary rounded-full transition-all duration-1000"
                  style={{ width: `${result.shap.xray * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-xs font-bold text-on-surface">
                <span className="text-secondary">Oxygen Saturation (SpO2)</span>
                <span className="text-primary">{result.shap.spo2}</span>
              </div>
              <div className="w-full h-2 bg-surface-container rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary rounded-full transition-all duration-1000"
                  style={{ width: `${result.shap.spo2 * 100}%` }}
                ></div>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-4">
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-bold text-on-surface">
                <span className="text-secondary">Respiratory Rate</span>
                <span className="text-primary">{result.shap.respRate}</span>
              </div>
              <div className="w-full h-2 bg-surface-container rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary rounded-full transition-all duration-1000"
                  style={{ width: `${result.shap.respRate * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-xs font-bold text-on-surface">
                <span className="text-secondary">Symptom Weight / Fever</span>
                <span className="text-primary">{result.shap.fever}</span>
              </div>
              <div className="w-full h-2 bg-surface-container rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary rounded-full transition-all duration-1000"
                  style={{ width: `${result.shap.fever * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
