import "dotenv/config";
import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOllama({
  model: process.env.OLLAMA_MODEL || "llama3.2:3b",
  baseUrl: "http://127.0.0.1:11434",
  temperature: 0.2
});

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `
You are a helpful AI assistant.

You answer questions about LangChain JS using ONLY the provided context.

Rules:
- If the answer exists in the context, answer clearly.
- If the answer does not exist in the context, say:
  "I don't have enough information in my knowledge base."
- Keep answers short and beginner-friendly.
- If the user asks in Arabic or Franco Arabic, answer in Arabic.
- If the user asks in English, answer in English.

Context:
{context}
`
  ],
  ["human", "{question}"]
]);

export const ragChain = prompt.pipe(model).pipe(new StringOutputParser());