import { GithubRepoLoader } from "@langchain/community/document_loaders/web/github";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import ignore from "ignore";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { similarity } from "ml-distance";

import { config } from "dotenv";
config();

const loader = new GithubRepoLoader(
    'https://github.com/langchain-ai/langchainjs',
    {recursive: false, ignorePaths: ['*.md','yarn.lock']},
);

const docs = await loader.load();

//console.log(docs.slice(0,3));

const splitter = RecursiveCharacterTextSplitter.fromLanguage('js', {
    chunkSize: 64,
    chunkOverlap: 0,
});

const splitDocs = await splitter.splitDocuments(docs);

//console.log(splitDocs.slice(0,3));

const loader2 = new PDFLoader('./MachineLearning-Lecture01.pdf')

const docs2 = await loader2.load();

//console.log(docs2.slice(0,3));

const pdfSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1536,
    chunkOverlap: 128,
})

const splitPdfDocs = await pdfSplitter.splitDocuments(docs2);

const embeddings = new OpenAIEmbeddings();

// const vector1 = await embeddings.embedQuery(
//     'what are vectors useful for in machine learning?'
// )

// const vector2 = await embeddings.embedQuery(
//     'A group of parrots is called a pandemonium.'
// );

// const similarVector = await embeddings.embedQuery(
//     'Vectors are represntation of information'
// );

// console.log('Less similar: ', similarity.cosine(vector1, vector2))
// console.log('More Similar: ', similarity.cosine(vector1, similarVector))

const vectorStore = new MemoryVectorStore(embeddings);

await vectorStore.addDocuments(splitPdfDocs);

const retriever = vectorStore.asRetriever();

import { RunnableSequence } from "@langchain/core/runnables";

const convertDocsToString = (documents) => {
  return documents.map((document) => {
    return `<doc>\n${document.pageContent}\n</doc>`
  }).join("\n");
};

const documentRetrievalChain = RunnableSequence.from([
    (input) => input.question,
    retriever,
    convertDocsToString
]);

// const result = await documentRetrievalChain.invoke({
// question: "What are the prerequisites for this course?"
// });

//console.log(result);


import { ChatPromptTemplate } from "@langchain/core/prompts";

const TEMPLATE_STRING = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the provided context, answer the user's question 
to the best of your ability using only the resources provided. 
Be verbose!

<context>

{context}

</context>

Now, answer this question using the above context:

{question}`;

const answerGenerationPrompt = ChatPromptTemplate.fromTemplate(
    TEMPLATE_STRING
);

import { RunnableMap } from "@langchain/core/runnables";

// const runnableMap = RunnableMap.from({
//   context: documentRetrievalChain,
//   question: (input) => input.question,
// });

// await runnableMap.invoke({
//     question: "What are the prerequisites for this course?"
// })

import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOpenAI({
    modelName: "gpt-4",
    temperature: 0.0,
});

const retrievalChain = RunnableSequence.from([
  {
    context: documentRetrievalChain,
    question: (input) => input.question,
  },
  answerGenerationPrompt,
  model,
  new StringOutputParser(),
]);

const answer = await retrievalChain.invoke({
  question: "What are the prerequisites for this course?"
});

console.log(answer);

// const followupAnswer = await retrievalChain.invoke({
//   question: "Can you list them in bullet point form?"
// });

// console.log(followupAnswer);

// const docs = await documentRetrievalChain.invoke({
//   question: "Can you list them in bullet point form?"
// });

// console.log(docs);

  import { MessagesPlaceholder } from "@langchain/core/prompts";

const REPHRASE_QUESTION_SYSTEM_TEMPLATE = 
  `Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.`;

const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human", 
    "Rephrase the following question as a standalone question:\n{question}"
  ],
]);