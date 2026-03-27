import express from "express";
import dotenv from "dotenv";
import OpenAI from "openai";
import { retrieveDocs } from "./rag/query.js";

dotenv.config();

const app = express();
app.use(express.json());

const client = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1"
});

const sessions = {};

app.post("/chat", async (req, res) => {
  try {
    const { message, userId } = req.body;

    if (!message || !userId) {
      return res.status(400).json({
        error: "message and userId are required"
      });
    }

    // create session if first message
    if (!sessions[userId]) {
      sessions[userId] = [
        {
          role: "system",
          content: "You are a helpful AI assistant. Use the provided context when answering questions."
        }
      ];
    }

    let chatHistory = sessions[userId];

    // add user message
    chatHistory.push({
      role: "user",
      content: message
    });

    // -------- RAG RETRIEVAL --------
    let context = "";
    try {
      context = await retrieveDocs(message);
    } catch (error) {
      console.log("RAG retrieval failed:", error);
    }

    // summarization if history too long
    if (chatHistory.length > 12) {
      try {

        const latestMessage = chatHistory[chatHistory.length - 1];

        const summaryResponse = await client.chat.completions.create({
          model: "llama-3.1-8b-instant",
          messages: [
            { role: "system", content: "Summarize this conversation briefly." },
            ...chatHistory.slice(0, -1)
          ]
        });

        const summaryContent =
          summaryResponse.choices[0].message.content;

        chatHistory = [
          {
            role: "system",
            content: "You are a helpful AI assistant."
          },
          {
            role: "system",
            content: `Summary of previous conversation: ${summaryContent}`
          },
          latestMessage
        ];

        sessions[userId] = chatHistory;

      } catch (error) {
        console.error("Summarization error:", error);
      }
    }
    console.log("RAG context:", context);


    // -------- MAIN LLM CALL --------

    const response = await client.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "system",
          content: `Use the following context to answer the user's question:\n${context}`
        },
        ...chatHistory
      ],
      temperature: 0.4,
      max_tokens: 200
    });

    const reply = response.choices[0].message.content;

    // store assistant response
    chatHistory.push({
      role: "assistant",
      content: reply
    });

    sessions[userId] = chatHistory;

    console.log("Chat history for", userId, chatHistory);

    res.json({ reply });

  } catch (error) {
    console.error("Server error:", error);
    res.status(500).json({
      error: "Internal server error"
    });
  }
});

app.listen(3000, () => {
  console.log("RAG Chatbot server running on port 3000");
});