import { Chroma } from "@langchain/community/vectorstores/chroma";
// The correct, exported path for modern LangChain versions
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

// This runs on your CPU, so no more "Waiting for Hugging Face" hangs!
const embeddings = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});

export async function retrieveDocs(question) {
  try {
    const vectorStore = await Chroma.fromExistingCollection(embeddings, {
      collectionName: "pdf-docs",
      url: "http://localhost:8000",
    });

    const retriever = vectorStore.asRetriever();
    const relevantDocs = await retriever.invoke(question);
    
    return relevantDocs.map(doc => doc.pageContent).join("\n\n");
  } catch (err) {
    console.error("RAG retrieval error:", err.message);
    return "";
  }
}