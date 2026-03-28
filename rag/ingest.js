import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import pdfParse from "pdf-parse-fork";
import "dotenv/config";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChromaClient } from "chromadb"; // Add this for a direct reset

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function ingest() {
  try {
    console.log("--- Starting Ingestion ---");
    
    // 1. Load PDF
    const pdfPath = path.join(__dirname, "../data/APAR_GUPTA_RESUME (8).pdf");
    const buffer = fs.readFileSync(pdfPath);
    const pdfData = await pdfParse(buffer);
    
    // 2. Split Text
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const docs = await splitter.createDocuments([pdfData.text]);
    console.log(`Created ${docs.length} chunks.`);

    // 3. Embeddings
    const embeddings = new HuggingFaceTransformersEmbeddings({
      model: "Xenova/all-MiniLM-L6-v2",
    });

    // 4. DIRECT RESET (This fixes the dimension error)
    console.log("Deleting old collection via ChromaClient...");
    const client = new ChromaClient({ path: "http://localhost:8000" });
    try {
        await client.deleteCollection({ name: "pdf-docs" });
        console.log("✅ Old collection wiped.");
    } catch (e) {
        console.log("ℹ️ Collection did not exist or already wiped.");
    }

    // 5. Store
    console.log("Storing new 384-dimension embeddings...");
    await Chroma.fromDocuments(docs, embeddings, {
      collectionName: "pdf-docs",
      url: "http://localhost:8000",
    });

    console.log(" SUCCESS: Documents stored in ChromaDB.");
  } catch (err) {
    console.error(" Error during ingestion:", err.message);
  }
}

ingest();