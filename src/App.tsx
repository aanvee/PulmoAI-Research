import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import Hero from "./components/Hero";
import DiagnosticInterface from "./components/DiagnosticInterface";
import AnalysisResults from "./components/AnalysisResults";
import ResearchMethodology from "./components/ResearchMethodology";
import Footer from "./components/Footer";
import { SAMPLE_CASES } from "./data";
import { AnalysisResult, PatientCase } from "./types";

export default function App() {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  
  // Track currently active images for display
  const [activeXrayUrl, setActiveXrayUrl] = useState<string>("");
  const [activeHeatmapUrl, setActiveHeatmapUrl] = useState<string>("");

  // Preload Case 1 (Pneumonia Suspect) as the active view on launch
  useEffect(() => {
    const defaultCase = SAMPLE_CASES[0];
    if (defaultCase) {
      setActiveXrayUrl(defaultCase.xrayUrl);
      setActiveHeatmapUrl(defaultCase.heatmapUrl);
      setAnalysisResult({
        probabilities: {
          pneumonia: defaultCase.pneumonia,
          pleuralEffusion: defaultCase.pleuralEffusion,
          pneumothorax: defaultCase.pneumothorax,
        },
        analysis: defaultCase.analysis,
        recommendations: defaultCase.recommendations,
        shap: defaultCase.shap,
        topPathology: "Pneumonia",
      });
    }
  }, []);

  const handleAnalyze = async (patientData: {
    age: string;
    gender: string;
    temp: string;
    heartRate: string;
    respRate: string;
    spO2: string;
    symptoms: string[];
    caseName: string;
    customImageUploaded: boolean;
    customImageBase64: string;
    selectedCase?: PatientCase;
  }) => {
    setIsLoading(true);

    try {
      // If a pre-defined sample case is active and no custom image uploaded, we can instantly
      // load its state (for sub-second response) or simulate a network/API fetch
      if (patientData.selectedCase && !patientData.customImageUploaded) {
        const selected = patientData.selectedCase;
        // Introduce a realistic sub-second processing lag to give professional weight to the analysis
        await new Promise((resolve) => setTimeout(resolve, 800));
        
        setActiveXrayUrl(selected.xrayUrl);
        setActiveHeatmapUrl(selected.heatmapUrl);
        setAnalysisResult({
          probabilities: {
            pneumonia: selected.pneumonia,
            pleuralEffusion: selected.pleuralEffusion,
            pneumothorax: selected.pneumothorax,
          },
          analysis: selected.analysis,
          recommendations: selected.recommendations,
          shap: selected.shap,
          topPathology: selected.pneumonia > selected.pleuralEffusion && selected.pneumonia > selected.pneumothorax 
            ? "Pneumonia" 
            : selected.pleuralEffusion > selected.pneumothorax 
            ? "Pleural Effusion" 
            : "Pneumothorax",
        });

        // Smooth scroll to results
        setTimeout(() => {
          document.getElementById("results-section")?.scrollIntoView({ behavior: "smooth" });
        }, 100);
      } else {
        // Perform real full-stack API prediction request to server-side Gemini
        const response = await fetch("/api/analyze", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(patientData),
        });

        if (!response.ok) {
          throw new Error("API analysis request failed");
        }

        const data = await response.json();
        
        if (data.success) {
          // If custom image uploaded, use it. Otherwise, fallback to base xrays
          if (patientData.customImageUploaded && patientData.customImageBase64) {
            setActiveXrayUrl(patientData.customImageBase64);
            setActiveHeatmapUrl(patientData.customImageBase64); // No visual heatmap available for custom unless generated
          } else {
            setActiveXrayUrl(SAMPLE_CASES[0].xrayUrl);
            setActiveHeatmapUrl(SAMPLE_CASES[0].heatmapUrl);
          }

          setAnalysisResult({
            probabilities: {
              pneumonia: data.probabilities.pneumonia,
              pleuralEffusion: data.probabilities.pleuralEffusion,
              pneumothorax: data.probabilities.pneumothorax,
            },
            analysis: data.analysis,
            recommendations: data.recommendations,
            shap: data.shap,
            topPathology: data.topPathology,
          });

          // Smooth scroll to results
          setTimeout(() => {
            document.getElementById("results-section")?.scrollIntoView({ behavior: "smooth" });
          }, 100);
        }
      }
    } catch (err) {
      console.error("Clinical analyzer error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-background min-h-screen text-on-surface font-sans selection:bg-primary-fixed select-text">
      {/* Header Navigation */}
      <Header />

      {/* Main Content Layout */}
      <main className="space-y-4">
        {/* Hero Banner with Action Scroll Button */}
        <Hero />

        {/* Diagnostic Form with prefill options and upload support */}
        <DiagnosticInterface onAnalyze={handleAnalyze} isLoading={isLoading} />

        {/* Dynamic Pathology calculations, Grad-CAM Saliency Maps and Gemini AI interpretation */}
        <AnalysisResults 
          result={analysisResult} 
          xrayUrl={activeXrayUrl} 
          heatmapUrl={activeHeatmapUrl} 
          isLoading={isLoading} 
        />

        {/* Technical overview of models, Streams, and SOTA validation metrics */}
        <ResearchMethodology />
      </main>

      {/* Institutional research disclaimer and tech stack links */}
      <Footer />
    </div>
  );
}
