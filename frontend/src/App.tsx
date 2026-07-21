import React, { useState } from "react";
import Header from "./components/Header";
import Hero from "./components/Hero";
import DiagnosticInterface from "./components/DiagnosticInterface";
import AnalysisResults from "./components/AnalysisResults";
import ResearchMethodology from "./components/ResearchMethodology";
import Footer from "./components/Footer";
import { AnalysisResult } from "./types";

export default function App() {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Track currently active images for display
  const [activeXrayUrl, setActiveXrayUrl] = useState<string>("");
  const [activeHeatmapUrl, setActiveHeatmapUrl] = useState<string>("");
  const [activeShapUrl, setActiveShapUrl] = useState<string>("");


  const handleAnalyze = async (patientData: {
    age: string;
    gender: string;
    temp: string;
    heartRate: string;
    respRate: string;
    spO2: string;
    symptoms: string[];
    customImageUploaded: boolean;
    customImageBase64: string;
    imageFile: File | null;
  }) => {
    setIsLoading(true);

    try {
      const genderCode = patientData.gender === "Male" ? "M" : patientData.gender === "Female" ? "F" : patientData.gender.toUpperCase() === "M" ? "M" : "F";

      // -------------------------
      // 1. Prediction Request
      // -------------------------

      console.log({
        age: patientData.age,
        gender: genderCode,
        view_position: "PA",
        image: patientData.imageFile
      });

      const formData = new FormData();

      formData.append("image", patientData.imageFile!);
      formData.append("age", patientData.age);
      formData.append("gender", genderCode);
      formData.append("view_position", "PA");

      const response = await fetch(
        "http://127.0.0.1:8000/api/predict/",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Prediction API Error:", errorData);
        throw new Error("Prediction API request failed.");
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error("Prediction failed.");
      }

      // Save prediction results and set original X-ray
      setAnalysisResult({
        predictions: data.predictions,
        top_predictions: data.top_predictions,
      });
      setActiveXrayUrl(patientData.customImageBase64);

      // -------------------------
      // 2. Grad-CAM Request
      // -------------------------
      const targetClass = data.top_predictions[0].disease;

      const gradcamForm = new FormData();
      gradcamForm.append("image", patientData.imageFile!);
      gradcamForm.append("age", patientData.age);
      gradcamForm.append("gender", genderCode);
      gradcamForm.append("view_position", "PA");
      gradcamForm.append("target_class", targetClass);

      const gradcamResponse = await fetch(
        "http://127.0.0.1:8000/api/gradcam/",
        {
          method: "POST",
          body: gradcamForm,
        }
      );

      if (!gradcamResponse.ok) {
        const errorData = await gradcamResponse.json();
        console.error("Grad-CAM API Error:", errorData);
        throw new Error("Grad-CAM request failed.");
      }

      const gradcamData = await gradcamResponse.json();

      if (!gradcamData.success) {
        throw new Error("Grad-CAM generation failed.");
      }

      setActiveHeatmapUrl(
        `http://127.0.0.1:8000${gradcamData.gradcam_image}`
      );

      // -------------------------
      // 3. SHAP Request
      // -------------------------
      const shapForm = new FormData();
      shapForm.append("age", patientData.age);
      shapForm.append("gender", genderCode);
      shapForm.append("view_position", "PA");

      const shapResponse = await fetch(
        "http://127.0.0.1:8000/api/shap/",
        {
          method: "POST",
          body: shapForm,
        }
      );

      if (!shapResponse.ok) {
        const errorData = await shapResponse.json();
        console.error("SHAP API Error:", errorData);
        throw new Error("SHAP request failed.");
      }

      const shapData = await shapResponse.json();

      if (!shapData.success) {
        throw new Error("SHAP generation failed.");
      }

      setActiveShapUrl(
        `http://127.0.0.1:8000${shapData.shap_plot}`
      );

      // Scroll to results
      document
        .getElementById("results-section")
        ?.scrollIntoView({
          behavior: "smooth",
        });

    } catch (error) {

      console.error(error);

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
          shapUrl={activeShapUrl}
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
