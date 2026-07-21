export interface DiseasePrediction {
  probability: number;
  prediction: boolean;
}

export interface TopPrediction {
  disease: string;
  probability: number;
}

export interface AnalysisResult {
  predictions: {
    [disease: string]: DiseasePrediction;
  };

  top_predictions: TopPrediction[];
}

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
  pneumonia?: number;
  covid?: number;
  atelectasis?: number;
  normal?: number;
  pleuralEffusion?: number;
  pneumothorax?: number;
  [key: string]: any;
}