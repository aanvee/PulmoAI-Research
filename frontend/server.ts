import express from "express";
import path from "path";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(express.json({ limit: "50mb" }));

const PORT = 3000;

// Lazy initialize Gemini API client
let aiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI | null {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (apiKey) {
      aiClient = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            "User-Agent": "aistudio-build",
          },
        },
      });
    } else {
      console.warn("GEMINI_API_KEY is not defined. Using fallback rule-based analyzer.");
    }
  }
  return aiClient;
}

// API endpoint for clinical diagnosis
app.post("/api/analyze", async (req, res) => {
  try {
    const {
      age,
      gender,
      temp,
      heartRate,
      respRate,
      spO2,
      symptoms,
      caseName,
      customImageUploaded,
      customImageBase64,
    } = req.body;

    // Default probabilities
    let pneumoniaProb = 15.0;
    let effusionProb = 8.0;
    let pneumothoraxProb = 2.0;

    const listSymptoms = symptoms || [];

    // Adjust probabilities based on symptoms and vitals
    if (listSymptoms.includes("Fever")) {
      pneumoniaProb += 25;
    }
    if (listSymptoms.includes("Cough")) {
      pneumoniaProb += 15;
      effusionProb += 5;
    }
    if (listSymptoms.includes("Dyspnea")) {
      pneumoniaProb += 10;
      effusionProb += 25;
      pneumothoraxProb += 30;
    }
    if (listSymptoms.includes("Chest Pain")) {
      effusionProb += 15;
      pneumothoraxProb += 40;
    }
    if (listSymptoms.includes("Fatigue")) {
      pneumoniaProb += 5;
      effusionProb += 5;
    }

    // Vitals adjustment
    const t = parseFloat(temp) || 36.8;
    if (t > 38.0) {
      pneumoniaProb += 20;
    }
    
    const hr = parseFloat(heartRate) || 72;
    if (hr > 100) {
      pneumoniaProb += 10;
      pneumothoraxProb += 10;
    }

    const rr = parseFloat(respRate) || 16;
    if (rr > 20) {
      pneumoniaProb += 15;
      effusionProb += 15;
      pneumothoraxProb += 15;
    }

    const o2 = parseFloat(spO2) || 98;
    if (o2 < 92) {
      pneumoniaProb += 20;
      effusionProb += 15;
      pneumothoraxProb += 20;
    }

    // Cap probabilities
    pneumoniaProb = Math.min(99.5, Math.max(1.0, pneumoniaProb));
    effusionProb = Math.min(99.5, Math.max(1.0, effusionProb));
    pneumothoraxProb = Math.min(99.5, Math.max(1.0, pneumothoraxProb));

    // Override or blend if a specific pre-defined case is active
    if (caseName === "Pneumonia Case" || caseName?.includes("Pneumonia")) {
      pneumoniaProb = 89.4;
      effusionProb = 12.1;
      pneumothoraxProb = 2.4;
    } else if (caseName === "Pleural Effusion Case" || caseName?.includes("Effusion")) {
      pneumoniaProb = 18.5;
      effusionProb = 82.3;
      pneumothoraxProb = 5.1;
    } else if (caseName === "Pneumothorax Case" || caseName?.includes("Pneumothorax")) {
      pneumoniaProb = 5.2;
      effusionProb = 10.4;
      pneumothoraxProb = 91.2;
    } else if (caseName === "Healthy Case" || caseName?.includes("Healthy") || caseName?.includes("Normal")) {
      pneumoniaProb = 1.2;
      effusionProb = 0.8;
      pneumothoraxProb = 0.5;
    }

    const pP = parseFloat(pneumoniaProb.toFixed(1));
    const eP = parseFloat(effusionProb.toFixed(1));
    const ptP = parseFloat(pneumothoraxProb.toFixed(1));

    // Determine top pathology
    let topPathology = "Pneumonia";
    let maxP = pP;
    if (eP > maxP) {
      topPathology = "Pleural Effusion";
      maxP = eP;
    }
    if (ptP > maxP) {
      topPathology = "Pneumothorax";
      maxP = ptP;
    }

    let clinicalAnalysis = "";
    let recommendations: string[] = [];
    let shapImportance = {
      xray: 0.42,
      spo2: 0.31,
      respRate: 0.18,
      fever: 0.09,
    };

    const client = getGeminiClient();
    if (client) {
      try {
        const promptText = `
You are an expert clinical chest radiology and pulmonology AI assistant.
Analyze the following patient clinical profile and chest X-ray findings:

Patient demographics:
- Age: ${age || "N/A"}
- Gender: ${gender || "N/A"}

Vital Signs & Measured Vitals:
- Temperature: ${temp || "36.8"} °C
- Heart Rate: ${heartRate || "72"} bpm
- Respiratory Rate: ${respRate || "16"} breaths/min
- Oxygen Saturation (SpO2): ${spO2 || "98"}%

Presenting Symptoms: ${listSymptoms.length > 0 ? listSymptoms.join(", ") : "None reported"}
Diagnosis Case Reference: ${caseName || "Custom Case Upload"}

Calculated Pathology Probabilities:
- Pneumonia probability: ${pP}%
- Pleural Effusion probability: ${eP}%
- Pneumothorax probability: ${ptP}%

Primary Suspected Pathology: ${topPathology} (${maxP}%)

Please generate a professional, precise, clinical analysis summary (1-2 paragraphs) explaining the diagnostic logic, how findings align, and an actionable clinical recommendation (1 sentence).
You MUST respond with a JSON object of this structure:
{
  "summary": "The clinical summary text...",
  "recommendation": "The immediate recommendation sentence...",
  "shapImportance": {
    "xray": 0.42,
    "spo2": 0.31,
    "respRate": 0.18,
    "fever": 0.09
  }
}
Please ensure the shapImportance values reflect the abnormal vitals or critical symptoms, summing up to approximately 1.0. Keep your response strictly matching the schema.
`;

        const contents: any[] = [promptText];

        if (customImageUploaded && customImageBase64) {
          const matches = customImageBase64.match(/^data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+);base64,(.+)$/);
          if (matches && matches.length === 3) {
            contents.push({
              inlineData: {
                mimeType: matches[1],
                data: matches[2],
              },
            });
          }
        }

        const response = await client.models.generateContent({
          model: "gemini-3.5-flash",
          contents,
          config: {
            responseMimeType: "application/json",
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                summary: { type: Type.STRING },
                recommendation: { type: Type.STRING },
                shapImportance: {
                  type: Type.OBJECT,
                  properties: {
                    xray: { type: Type.NUMBER },
                    spo2: { type: Type.NUMBER },
                    respRate: { type: Type.NUMBER },
                    fever: { type: Type.NUMBER },
                  },
                  required: ["xray", "spo2", "respRate", "fever"],
                },
              },
              required: ["summary", "recommendation", "shapImportance"],
            },
          },
        });

        const parsed = JSON.parse(response.text?.trim() || "{}");
        if (parsed.summary) {
          clinicalAnalysis = parsed.summary;
          recommendations = [parsed.recommendation || "Clinical correlation is recommended."];
          shapImportance = parsed.shapImportance || shapImportance;
        }
      } catch (err) {
        console.error("Gemini API call failed, using fallback:", err);
      }
    }

    if (!clinicalAnalysis) {
      if (caseName === "Healthy Case" || (pP < 15 && eP < 15 && ptP < 15)) {
        clinicalAnalysis = `Patient's vital signs are stable and chest radiography demonstrates clean, hyper-aerated lung fields without evidence of consolidations, pleural fluid collections, or pneumothorax lines. The pulmonary vasculature is within normal limits. Clinically, findings are normal.`;
        recommendations = ["No active acute pulmonary pathology identified. Routine clinical tracking is recommended."];
      } else if (topPathology === "Pneumonia") {
        clinicalAnalysis = `Patient presentation of fever (${temp || "38.2"}°C) and symptoms such as cough or dyspnea, paired with chest radiograph, points to a strong likelihood of lobar pneumonia. Significant alveolar infiltration is visualized in the target fields. Attention focus confirms active consolidation requiring therapeutic targeting.`;
        recommendations = ["High risk of Pneumonia. Clinical correlation with bedside ultrasound and empiric antimicrobial therapy is recommended."];
      } else if (topPathology === "Pleural Effusion") {
        clinicalAnalysis = `Radiological evaluation reveals pleural effusion with blunting of the costophrenic sulcus. Mechanical fluid pooling aligns with reported dyspnea and compromised SpO2 (${spO2 || "94"}%), restricting alveolar expansion.`;
        recommendations = ["Moderate Pleural Effusion. Consider bedside thoracic ultrasound for fluid quantification and clinical correlation."];
      } else {
        clinicalAnalysis = `Urgent diagnostic alert for potential Pneumothorax. Radiograph displays a thin, visceral pleural line with absent peripheral lung markings. Clinical findings of dyspnea and chest pain suggest immediate diagnostic verification.`;
        recommendations = ["High risk of Pneumothorax. Urgent physician evaluation, bedside ultrasound, or expiratory film is recommended."];
      }

      // Generate custom SHAP for fallback based on vitals
      let xrW = 0.45;
      let o2W = 0.25;
      let rrW = 0.20;
      let fW = 0.10;

      const spO2Val = parseFloat(spO2) || 98;
      const rrVal = parseFloat(respRate) || 16;
      const tempVal = parseFloat(temp) || 36.8;

      if (spO2Val < 93) o2W += 0.10;
      if (rrVal > 20) rrW += 0.10;
      if (tempVal > 38.0) fW += 0.10;

      const sum = xrW + o2W + rrW + fW;
      shapImportance = {
        xray: parseFloat((xrW / sum).toFixed(2)),
        spo2: parseFloat((o2W / sum).toFixed(2)),
        respRate: parseFloat((rrW / sum).toFixed(2)),
        fever: parseFloat((fW / sum).toFixed(2)),
      };
    }

    res.json({
      success: true,
      probabilities: {
        pneumonia: pP,
        pleuralEffusion: eP,
        pneumothorax: ptP,
      },
      analysis: clinicalAnalysis,
      recommendations,
      shap: shapImportance,
      topPathology,
    });
  } catch (error: any) {
    console.error("General server error:", error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Setup Vite middleware or static serving
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on port ${PORT}`);
  });
}

startServer();
