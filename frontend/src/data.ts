import { PatientCase } from "./types";

export const XRAY_HOTLINK_URL = "https://lh3.googleusercontent.com/aida-public/AB6AXuBxN2grYRceSNkgjEoLfzbiYmAyBhHVnIScyzipdypd1J6eLrE0qSmgD3BSxpNLgmtnuPan35TTTLhcj5MZJ2OE1gII1voho8-UPJYa70DUnbMkrAsxtPLZtkTNpdzAiEvLn5jNVykrSMfjFmGTo_sVmK_3yVn2GnbxwXNzXgUYVuDeF3BrFGqxzRUpxdYNwZ3GwIB8kITKJwjR0Sew-J8tzg2CngWP4WPf21WFyniC6L5VFLQC16-J9g";
export const HEATMAP_HOTLINK_URL = "https://lh3.googleusercontent.com/aida-public/AB6AXuAPCj4V2aL1ARrX9bJrPAcwGU3qg7NSBClNknLQJuRnpHWSgQkW7GORtUkXfjEobLmYsYbck1y4YxXaVFFEAegZL9RLKPTUApKHfx9eXud73YC6KDX2tA4HcIxlU6hGZdqEoId1hSaK5LdJBrwGTWgNIjyd7lcTBL-wfHpPIdqkRD22d823WdTS07xGRCB9fBgQSFNbUtThzE3gNaLKzcargd2c7vBqLehLaHX5-Zq1-QJIzBdbeJWZAw";

export const SAMPLE_CASES: PatientCase[] = [
  {
    id: "pneumonia-case",
    name: "Pneumonia Suspect (Case 1)",
    description: "Elderly patient presenting with acute fever, productive cough, and shortness of breath.",
    age: "68",
    gender: "Male",
    temp: "38.5",
    heartRate: "94",
    respRate: "22",
    spO2: "91",
    symptoms: ["Fever", "Cough", "Dyspnea"],
    xrayUrl: XRAY_HOTLINK_URL,
    heatmapUrl: HEATMAP_HOTLINK_URL,
    pneumonia: 89.4,
    pleuralEffusion: 12.1,
    pneumothorax: 2.4,
    shap: {
      xray: 0.42,
      spo2: 0.31,
      respRate: 0.18,
      fever: 0.09
    },
    analysis: "Patient presents with signs heavily indicative of acute lobar pneumonia. Visual inspection reveals patchy focal consolidations in the right middle and lower lobes. Clinical correlation shows high fever (38.5°C) and tachypnea, reinforcing diagnostic certainty of a bacterial or viral respiratory infection.",
    recommendations: ["High risk detected. Clinical correlation with bedside ultrasound and empiric antimicrobial therapy is recommended."]
  },
  {
    id: "effusion-case",
    name: "Pleural Effusion (Case 2)",
    description: "Middle-aged patient reporting gradual onset shortness of breath and dull chest pain.",
    age: "52",
    gender: "Female",
    temp: "36.9",
    heartRate: "82",
    respRate: "20",
    spO2: "93",
    symptoms: ["Dyspnea", "Chest Pain"],
    xrayUrl: XRAY_HOTLINK_URL,
    heatmapUrl: HEATMAP_HOTLINK_URL,
    pneumonia: 18.5,
    pleuralEffusion: 82.3,
    pneumothorax: 5.1,
    shap: {
      xray: 0.38,
      spo2: 0.32,
      respRate: 0.20,
      fever: 0.10
    },
    analysis: "Radiological evaluation reveals pleural effusion with significant blunting of the right costophrenic sulcus. Mechanical fluid pooling aligns with reported dyspnea and compromised SpO2 (93%), restricting alveolar expansion and diaphragm movement.",
    recommendations: ["Moderate Pleural Effusion. Bedside thoracic ultrasound is recommended for fluid volume quantification and therapeutic thoracocentesis targeting."]
  },
  {
    id: "pneumothorax-case",
    name: "Pneumothorax Suspect (Case 3)",
    description: "Young adult presenting with sudden onset sharp chest pain, tachypnea, and dry cough.",
    age: "29",
    gender: "Male",
    temp: "36.6",
    heartRate: "105",
    respRate: "24",
    spO2: "92",
    symptoms: ["Cough", "Dyspnea", "Chest Pain"],
    xrayUrl: XRAY_HOTLINK_URL,
    heatmapUrl: HEATMAP_HOTLINK_URL,
    pneumonia: 5.2,
    pleuralEffusion: 10.4,
    pneumothorax: 91.2,
    shap: {
      xray: 0.50,
      spo2: 0.25,
      respRate: 0.18,
      fever: 0.07
    },
    analysis: "Urgent diagnostic alert for potential Pneumothorax. Radiograph displays a thin, visceral pleural line with absent peripheral lung markings in the apex. Clinical findings of tachycardia and acute pleuritic pain suggest immediate diagnostic verification.",
    recommendations: ["Immediate clinical correlation and needle decompression or small-bore chest tube placement should be evaluated by the physician."]
  },
  {
    id: "healthy-case",
    name: "Normal Control (Case 4)",
    description: "Pre-operative screening, completely asymptomatic with stable health indicators.",
    age: "41",
    gender: "Female",
    temp: "36.7",
    heartRate: "70",
    respRate: "14",
    spO2: "99",
    symptoms: [],
    xrayUrl: XRAY_HOTLINK_URL,
    heatmapUrl: XRAY_HOTLINK_URL, // Reuse chest X-ray but no highlight
    pneumonia: 1.2,
    pleuralEffusion: 0.8,
    pneumothorax: 0.5,
    shap: {
      xray: 0.25,
      spo2: 0.25,
      respRate: 0.25,
      fever: 0.25
    },
    analysis: "Patient's vital signs are stable and chest radiography demonstrates clean, hyper-aerated lung fields without evidence of consolidations, pleural fluid collections, or pneumothorax lines. The pulmonary vasculature is within normal physiological limits.",
    recommendations: ["No active pulmonary pathology identified. Routine clinical follow-up as indicated."]
  }
];
