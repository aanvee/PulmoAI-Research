import React, { useRef, useState } from "react";
import { Upload, ChevronDown, Check, RefreshCw } from "lucide-react";

interface DiagnosticInterfaceProps {
  onAnalyze: (patientData: {
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
  }) => void;
  isLoading: boolean;
}

export default function DiagnosticInterface({ onAnalyze, isLoading }: DiagnosticInterfaceProps) {
  const [age, setAge] = useState<string>("68");
  const [gender, setGender] = useState<string>("Male");
  const [temp, setTemp] = useState<string>("38.5");
  const [heartRate, setHeartRate] = useState<string>("94");
  const [respRate, setRespRate] = useState<string>("22");
  const [spO2, setSpO2] = useState<string>("91");
  const [symptoms, setSymptoms] = useState<string[]>(["Fever", "Cough", "Dyspnea"]);

  // Custom upload state
  const [customImage, setCustomImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [customImageName, setCustomImageName] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleToggleSymptom = (symptom: string) => {
    if (symptoms.includes(symptom)) {
      setSymptoms(symptoms.filter((s) => s !== symptom));
    } else {
      setSymptoms([...symptoms, symptom]);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];

    if (!file) return;

    setImageFile(file);
    setCustomImageName(file.name);

    const reader = new FileReader();

    reader.onload = () => {
      setCustomImage(reader.result as string);
    };

    reader.readAsDataURL(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();

    const file = e.dataTransfer.files?.[0];

    if (!file) return;

    setImageFile(file);
    setCustomImageName(file.name);

    const reader = new FileReader();

    reader.onload = () => {
      setCustomImage(reader.result as string);
    };

    reader.readAsDataURL(file);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!imageFile) {
      alert("Please upload a Chest X-ray image.");
      return;
    }
    onAnalyze({
      age,
      gender,
      temp,
      heartRate,
      respRate,
      spO2,
      symptoms,
      customImageUploaded: !!customImage,
      customImageBase64: customImage || "",
      imageFile,
    });
  };

  const allAvailableSymptoms = ["Fever", "Cough", "Dyspnea", "Chest Pain", "Fatigue"];

  return (
    <section id="prediction" className="max-w-7xl mx-auto px-6 md:px-12 py-12 space-y-8 scroll-mt-20">
      {/* Title */}
      <div className="flex items-center gap-3">
        <div className="h-8 w-2 bg-primary rounded-full"></div>
        <h2 className="font-heading text-2xl sm:text-3xl font-bold text-on-surface">AI Diagnostic Interface</h2>
      </div>

      <form onSubmit={handleSubmit} className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left column: Image Upload */}
        <div className="lg:col-span-5 bg-surface-container-lowest rounded-3xl p-6 shadow-soft border border-outline-variant flex flex-col justify-between">
          <div>
            <h3 className="font-heading text-lg font-bold mb-4 flex items-center gap-2 text-on-surface">
              <Upload className="w-5 h-5 text-primary" />
              Chest X-Ray Imaging
            </h3>

            <div
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-outline-variant rounded-2xl flex flex-col items-center justify-center p-8 hover:border-primary hover:bg-primary/5 transition-all cursor-pointer group min-h-[250px]"
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".png,.jpg,.jpeg,.dcm"
                className="hidden"
              />

              {customImage ? (
                <div className="text-center space-y-3">
                  <div className="relative mx-auto w-24 h-24 rounded-lg overflow-hidden border border-outline">
                    <img src={customImage} alt="Preview" className="w-full h-full object-cover" />
                  </div>
                  <p className="font-sans text-sm font-semibold text-primary truncate max-w-[250px]">
                    {customImageName}
                  </p>
                  <p className="text-xs text-secondary">Click to upload a different file</p>
                </div>
              ) : (
                <div className="text-center">
                  <div className="w-14 h-14 rounded-full bg-primary-fixed flex items-center justify-center mb-4 mx-auto group-hover:scale-110 transition-transform">
                    <Upload className="w-6 h-6 text-primary" />
                  </div>
                  <p className="font-sans text-sm font-semibold text-on-surface">
                    Drag & drop chest X-ray DICOM/PNG
                  </p>
                  <p className="text-xs text-secondary mt-1">
                    Support for AP, PA, and Lateral views
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 p-4 bg-surface-container rounded-xl">
            <div className="flex items-center justify-between text-xs font-semibold text-on-surface-variant">
              <span>DICOM Metadata Check</span>
              <span className="text-primary font-bold tracking-wider flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-primary animate-ping"></span>
                READY
              </span>
            </div>
          </div>
        </div>

        {/* Right column: Patient Clinical Profile */}
        <div className="lg:col-span-7 bg-surface-container-lowest rounded-3xl p-6 shadow-soft border border-outline-variant flex flex-col justify-between">
          <div>
            <h3 className="font-heading text-lg font-bold mb-4 flex items-center gap-2 text-on-surface">
              <span className="p-1 rounded bg-primary/10 text-primary">🩺</span>
              Patient Clinical Profile
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-on-surface-variant">Age (Years)</label>
                <input
                  type="number"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  className="w-full bg-surface-container-low border-0 outline-none focus:ring-2 focus:ring-primary/40 rounded-xl h-12 px-4 text-sm font-medium transition-shadow"
                  placeholder="e.g. 45"
                  required
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-on-surface-variant">Gender</label>
                <div className="relative">
                  <select
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    className="w-full bg-surface-container-low border-0 outline-none focus:ring-2 focus:ring-primary/40 rounded-xl h-12 px-4 text-sm font-medium transition-shadow appearance-none cursor-pointer"
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                  <ChevronDown className="w-4 h-4 text-secondary absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none" />
                </div>
              </div>

              {/* Vitals grid container */}
              <div className="col-span-full grid grid-cols-2 md:grid-cols-4 gap-3 p-4 bg-surface-container-low rounded-2xl">
                <div className="space-y-1">
                  <label className="text-xs font-semibold text-outline">Temp (°C)</label>
                  <input
                    type="text"
                    value={temp}
                    onChange={(e) => setTemp(e.target.value)}
                    className="w-full bg-surface-container-lowest border-0 outline-none focus:ring-1 focus:ring-primary/40 rounded-lg h-9 px-3 text-xs font-bold"
                    placeholder="36.8"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-semibold text-outline">Heart Rate (bpm)</label>
                  <input
                    type="text"
                    value={heartRate}
                    onChange={(e) => setHeartRate(e.target.value)}
                    className="w-full bg-surface-container-lowest border-0 outline-none focus:ring-1 focus:ring-primary/40 rounded-lg h-9 px-3 text-xs font-bold"
                    placeholder="72"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-semibold text-outline">Resp Rate (breaths/min)</label>
                  <input
                    type="text"
                    value={respRate}
                    onChange={(e) => setRespRate(e.target.value)}
                    className="w-full bg-surface-container-lowest border-0 outline-none focus:ring-1 focus:ring-primary/40 rounded-lg h-9 px-3 text-xs font-bold"
                    placeholder="16"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-semibold text-outline">SpO2 (%)</label>
                  <input
                    type="text"
                    value={spO2}
                    onChange={(e) => setSpO2(e.target.value)}
                    className="w-full bg-surface-container-lowest border-0 outline-none focus:ring-1 focus:ring-primary/40 rounded-lg h-9 px-3 text-xs font-bold"
                    placeholder="98"
                  />
                </div>
              </div>

              {/* Presenting Symptoms checkboxes */}
              <div className="col-span-full space-y-2">
                <label className="text-xs font-semibold text-on-surface-variant block">Presenting Symptoms</label>
                <div className="flex flex-wrap gap-2">
                  {allAvailableSymptoms.map((symptom) => {
                    const isChecked = symptoms.includes(symptom);
                    return (
                      <button
                        key={symptom}
                        type="button"
                        onClick={() => handleToggleSymptom(symptom)}
                        className={`flex items-center gap-1.5 px-4 py-2 rounded-full border text-xs font-semibold transition-all cursor-pointer ${isChecked
                          ? "bg-primary-container text-on-primary-container border-primary"
                          : "bg-surface-container-lowest text-on-surface-variant border-outline-variant hover:border-primary"
                          }`}
                      >
                        {isChecked && <Check className="w-3.5 h-3.5" />}
                        {symptom}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className={`w-full mt-6 text-on-primary h-14 rounded-2xl font-bold text-base shadow-lg transition-all active:scale-[0.98] flex items-center justify-center gap-2 cursor-pointer ${isLoading
              ? "bg-primary/50 cursor-not-allowed"
              : "bg-primary hover:bg-primary-container hover:shadow-xl"
              }`}
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin" />
                Analyzing Multimodal Input...
              </>
            ) : (
              <>
                <span className="text-xl">📊</span>
                Analyze Patient
              </>
            )}
          </button>
        </div>
      </form>
    </section>
  );
}
