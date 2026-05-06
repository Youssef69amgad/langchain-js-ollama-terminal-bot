import "dotenv/config";
import readlineSync from "readline-sync";
import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

// ===============================
// 1. Knowledge Base
// ===============================

const knowledgeBase = [
  {
    id: 1,
    title: "What is LangChain?",
    content:
      "LangChain JS is a JavaScript framework used to build applications powered by Large Language Models. It helps developers connect LLMs with prompts, tools, memory, chains, agents, and external data sources."
  },
  {
    id: 2,
    title: "Why use LangChain?",
    content:
      "LangChain makes AI applications easier to organize. Instead of writing simple API calls only, developers can create structured workflows using prompts, models, tools, chains, retrievers, and agents."
  },
  {
    id: 3,
    title: "Prompt Template",
    content:
      "A prompt template is a reusable message structure. It allows developers to insert dynamic values into prompts before sending them to the language model."
  },
  {
    id: 4,
    title: "Chain",
    content:
      "A chain is a sequence of steps where the output of one step can become the input of the next step. In LangChain, chains help organize AI workflows."
  },
  {
    id: 5,
    title: "RAG",
    content:
      "RAG stands for Retrieval-Augmented Generation. It means retrieving relevant information from documents or a database before asking the language model to generate an answer."
  },
  {
    id: 6,
    title: "Agent",
    content:
      "An agent is an AI system that can decide what action to take. It can use tools, call functions, search data, or perform tasks depending on the user's request."
  },
  {
    id: 7,
    title: "Memory",
    content:
      "Memory allows an AI application to remember previous messages in a conversation. This makes the chatbot more conversational and context-aware."
  },
  {
    id: 8,
    title: "Tools",
    content:
      "Tools are functions that an AI agent can use to perform actions, such as searching the web, calling an API, calculating values, or retrieving information."
  },
  {
    id: 9,
    title: "Use Cases",
    content:
      "LangChain can be used to build chatbots, document question-answering systems, AI agents, customer support assistants, code assistants, and automation tools."
  }
];

// ===============================
// 2. Retriever
// ===============================

function normalizeText(text) {
  const stopWords = new Set([
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "and",
    "or",
    "can",
    "we",
    "using",
    "use",
    "does",
    "do",
    "with",
    "for"
  ]);

  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((word) => word.length > 2 && !stopWords.has(word));
}

function retrieveRelevantDocs(question, limit = 3) {
  const questionWords = normalizeText(question);

  const scoredDocs = knowledgeBase.map((doc) => {
    const docText = `${doc.title} ${doc.content}`.toLowerCase();

    let score = 0;

    for (const word of questionWords) {
      if (docText.includes(word)) {
        score++;
      }
    }

    return {
      ...doc,
      score
    };
  });

  return scoredDocs
    .filter((doc) => doc.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

function formatDocs(docs) {
  if (docs.length === 0) {
    return "No relevant context found.";
  }

  return docs
    .map((doc, index) => {
      return `
Document ${index + 1}
Title: ${doc.title}
Content: ${doc.content}
`;
    })
    .join("\n");
}

// ===============================
// 3. LangChain Model with Ollama
// ===============================

const model = new ChatOllama({
  model: process.env.OLLAMA_MODEL || "llama3.2:3b",
  baseUrl: process.env.OLLAMA_BASE_URL || "http://127.0.0.1:11434",
  temperature: 0.2
});

// ===============================
// 4. Prompt Template
// ===============================

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `
You are a helpful AI assistant.

You answer questions about LangChain JS using ONLY the provided context.

Rules:
- If the answer exists in the context, answer clearly.
- If the answer does not exist in the context, say exactly:
  "I don't have enough information in my knowledge base."
- Keep answers short and beginner-friendly.
- Always answer in English.
- Do not translate technical terms incorrectly.
- Explain technical terms simply.

Context:
{context}
`
  ],
  ["human", "{question}"]
]);

const chain = prompt.pipe(model).pipe(new StringOutputParser());

// ===============================
// 5. Terminal App
// ===============================

async function askBot(question) {
  const docs = retrieveRelevantDocs(question);
  const context = formatDocs(docs);

  console.log("\nRetrieved Documents:");
  console.log("--------------------");

  if (docs.length === 0) {
    console.log("No matching documents found.");
  } else {
    docs.forEach((doc) => {
      console.log(`- ${doc.title} | score: ${doc.score}`);
    });
  }

  const answer = await chain.invoke({
    context,
    question
  });

  return answer;
}

async function main() {
  console.log("======================================");
  console.log(" LangChain JS Terminal Bot - Ollama");
  console.log("======================================");
  console.log(`Using model: ${process.env.OLLAMA_MODEL || "llama3.2:3b"}`);
  console.log(`Ollama URL: ${process.env.OLLAMA_BASE_URL || "http://127.0.0.1:11434"}`);
  console.log("Ask anything about LangChain JS.");
  console.log("Type 'exit' to stop.");
  console.log("");

  while (true) {
    const question = readlineSync.question("You: ");

    if (!question.trim()) {
      console.log("Bot: Please enter a question.\n");
      continue;
    }

    if (question.toLowerCase() === "exit") {
      console.log("Bot: Goodbye!");
      break;
    }

    try {
      const answer = await askBot(question);

      console.log("\nBot:");
      console.log(answer);
      console.log("\n--------------------------------------\n");
    } catch (error) {
      console.error("\nFull Error:");
      console.error(error);

      console.log("\nPossible fixes:");
      console.log("1. Make sure Ollama is running:");
      console.log('   & "C:\\Users\\hp\\AppData\\Local\\Programs\\Ollama\\ollama.exe" serve');
      console.log("2. Make sure the model is installed:");
      console.log('   & "C:\\Users\\hp\\AppData\\Local\\Programs\\Ollama\\ollama.exe" pull llama3.2:3b');
      console.log("3. Make sure .env contains:");
      console.log("   OLLAMA_MODEL=llama3.2:3b");
      console.log("   OLLAMA_BASE_URL=http://127.0.0.1:11434");
      console.log("");
    }
  }
}

main();