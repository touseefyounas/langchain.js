import { config } from "dotenv";
config(); 
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { 
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
} 
from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";


const model = new ChatOpenAI({
  modelName: "gpt-4",
  temperature: 0.0,
});

// const message = await model.invoke([
//   new HumanMessage("Tell me a joke."),
// ]);

// console.log(message);

// const prompt = ChatPromptTemplate.fromTemplate(
//     `What are three good names for a compay that makes {product}?`
// )
// console.log(await prompt.invoke({
//   product: "colorful socks",
// }));

// const promptFromMessages = ChatPromptTemplate.fromMessages([
//     SystemMessagePromptTemplate.fromTemplate(
//         "You are an expert at picking company names."
//     ),
//     HumanMessagePromptTemplate.fromTemplate(
//         `What are three good names for a company that makes {product}?`
//     ),
// ]);

// console.log(await promptFromMessages.invoke({
//   product: "colorful socks",
// }));

const promptFromMessages = ChatPromptTemplate.fromMessages([
    ['system', 'You are an expert at picking company names.'],
    ['human', 'What are three good names for a company that makes {product}?'],
]);

// console.log(await promptFromMessages.invoke({
//   product: "colorful socks",
// }));


//LCEL
// LanfChain Expression Language

const outputParser = new StringOutputParser();
const chain = promptFromMessages.pipe(model).pipe(outputParser);

// console.log(await chain.invoke({
//   product: "autonomous drones",
// }));

const stream = await chain.stream({
    product: "autonomous drones",
});

for await (const chunk of stream) {
    console.log(chunk);
}