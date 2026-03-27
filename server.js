import express from "express";
import dotenv from "dotenv";
import OpenAI from "openai";
import cors from "cors"; // Recommended to avoid browser blocks
import { retrieveDocs } from "./rag/query.js";

dotenv.config();

const app = express();
app.use(cors()); // Enable CORS for future frontend use
app.use(express.json());

const client = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1"
});

const sessions = {};

// --- GUARDRAIL PROMPT ---
const SYSTEM_PERSONA = `
You are a Professional Document Assistant. 
Your expertise is strictly limited to the information provided in the Context below.

RULES:
1. Use ONLY the provided Context to answer the user's question.
2. If the answer is not found in the Context, say: "I'm sorry, I don't have information about that in the uploaded documents."
3. Never make up facts or use outside knowledge (hallucinate).
4. Do not answer questions about politics, general trivia, or unrelated topics. Politely redirect the user to ask about the documents.
5. Keep your tone helpful, concise, and professional.
`;

app.post("/chat", async (req, res) => {
  try {
    const { message, userId } = req.body;

    if (!message || !userId) {
      return res.status(400).json({ error: "message and userId are required" });
    }

    // Initialize session
    if (!sessions[userId]) {
      sessions[userId] = [];
    }

    let chatHistory = sessions[userId];

    // -------- RAG RETRIEVAL --------
    let context = "";
    try {
      context = await retrieveDocs(message);
    } catch (error) {
      console.log("RAG retrieval failed:", error.message);
    }

    // Summarization logic (keeps the history manageable)
    if (chatHistory.length > 10) {
      try {
        const summaryResponse = await client.chat.completions.create({
          model: "llama-3.1-8b-instant",
          messages: [
            { role: "system", content: "Summarize this conversation briefly so far." },
            ...chatHistory
          ]
        });
        
        const summaryContent = summaryResponse.choices[0].message.content;
        
        // Reset history with the summary
        chatHistory = [
          { role: "system", content: `Summary of previous talk: ${summaryContent}` }
        ];
      } catch (error) {
        console.error("Summarization error:", error);
      }
    }

    // -------- MAIN LLM CALL WITH GUARDRAILS --------
    const response = await client.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "system",
          content: `${SYSTEM_PERSONA}\n\n--- CONTEXT FROM DOCUMENTS ---\n${context || "No context found."}`
        },
        ...chatHistory,
        { role: "user", content: message }
      ],
      temperature: 0.1, // LOW temperature is critical for guardrails (avoids creativity)
      max_tokens: 300
    });

    const reply = response.choices[0].message.content;

    // Update history
    chatHistory.push({ role: "user", content: message });
    chatHistory.push({ role: "assistant", content: reply });
    sessions[userId] = chatHistory;

    res.json({ reply });

  } catch (error) {
    console.error("Server error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(3000, () => {
  console.log("🚀 Professional RAG Server running on port 3000");
});