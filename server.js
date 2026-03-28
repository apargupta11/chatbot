import express from "express";
import dotenv from "dotenv";
import OpenAI from "openai";
import cors from "cors";
import { MongoClient, ObjectId } from "mongodb";
import { retrieveDocs } from "./rag/query.js";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const client = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1"
});

let db;
const mongoClient = new MongoClient(process.env.MONGODB_URI);

const SYSTEM_PERSONA = `
You are a Professional Data Assistant. 
You have access to a Movie Database (MongoDB) and a Resume PDF (ChromaDB).
RULES:
1. If the user asks about themselves, use the 'User/Commenter' data.
2. If the user asks about a movie, use the 'Movie' data.
3. If they ask about professional experience, use the 'PDF Context'.
4. If you don't know the answer, say you don't know.
`;

app.post("/chat", async (req, res) => {
  try {
    const { message, userId } = req.body;

    if (!message || !userId || !ObjectId.isValid(userId)) {
      return res.status(400).json({ error: "Valid message and 24-char userId are required" });
    }

    const objId = new ObjectId(userId);

    // --- SMART SEARCH (Option 2) ---
    // We look in all 3 major collections at the same time
    const [commentData, movieData, userAccount, userSession] = await Promise.all([
      db.collection("comments").findOne({ _id: objId }),
      db.collection("movies").findOne({ _id: objId }),
      db.collection("users").findOne({ _id: objId }),
      db.collection("chat_histories").findOne({ userId: userId })
    ]);

    // Format the Database Context based on what we found
    let dbContext = "No specific database record found for this ID.";
    
    if (movieData) {
      dbContext = `MOVIE FOUND: "${movieData.title}" (${movieData.year}). Plot: ${movieData.fullplot || movieData.plot}`;
    } else if (commentData) {
      dbContext = `USER/COMMENTER FOUND: Name: ${commentData.name}, Email: ${commentData.email}, Last Comment: ${commentData.text}`;
    } else if (userAccount) {
      dbContext = `ACCOUNT HOLDER FOUND: Name: ${userAccount.name}, Email: ${userAccount.email}`;
    }

    // --- RAG RETRIEVAL (PDF) ---
    let pdfContext = "";
    try {
      pdfContext = await retrieveDocs(message);
    } catch (err) {
      console.log("PDF Search skipped.");
    }

    // --- MEMORY & RESPONSE ---
    let history = userSession?.messages || [];
    let runningSummary = userSession?.summary || "";

    const response = await client.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "system",
          content: `${SYSTEM_PERSONA}\nDATABASE CONTEXT: ${dbContext}\nPDF CONTEXT: ${pdfContext}\nSUMMARY: ${runningSummary}`
        },
        ...history,
        { role: "user", content: message }
      ],
      temperature: 0.1
    });

    const reply = response.choices[0].message.content;

    // --- UPDATE MEMORY ---
    await db.collection("chat_histories").updateOne(
      { userId: userId },
      {
        $set: { lastActive: new Date() },
        $push: {
          messages: {
            $each: [{ role: "user", content: message }, { role: "assistant", content: reply }],
            $slice: -10 
          }
        }
      },
      { upsert: true }
    );

    res.json({ reply });

  } catch (error) {
    console.error("Server error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// --- STARTUP ---
const startServer = async () => {
  try {
    await mongoClient.connect();
    db = mongoClient.db("sample_mflix");
    console.log("✅ Universal MongoDB Connected");
    app.listen(3000, () => console.log("🚀 Server running on port 3000"));
  } catch (error) {
    console.error("❌ Failed to start:", error);
  }
};

startServer();