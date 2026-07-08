export interface PatientCase {
  id: string;
  name: string;
  description: string;
  age: string;
  gender: string;
  temp: string;
  heartRate: string;
  respRate: string;
  spO2: string;
  symptoms: string[];
  xrayUrl: string;
  heatmapUrl: string;
  pneumonia: number;
  pleuralEffusion: number;
  pneumothorax: number;
  shap: {
    xray: number;
    spo2: number;
    respRate: number;
    fever: number;
  };
  analysis: string;
  recommendations: string[];
}

export interface AnalysisResult {
  probabilities: {
    pneumonia: number;
    pleuralEffusion: number;
    pneumothorax: number;
  };
  analysis: string;
  recommendations: string[];
  shap: {
    xray: number;
    spo2: number;
    respRate: number;
    fever: number;
  };
  topPathology: string;
}
