import os
import asyncio
import fitz  # PyMuPDF for PDF extraction
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up the Telegram bot
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

pdf_text = extract_text_from_pdf("Teknikavtalet_IF_Metall.pdf")

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=300)
chunks = text_splitter.split_text(pdf_text)

# Convert text chunks to Document objects
documents = [Document(page_content=chunk) for chunk in chunks]

# Create a vector store for retrieval
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
vector_store = InMemoryVectorStore.from_documents(documents, embeddings_model)
retriever = vector_store.as_retriever()

# Define the prompt template
template = """
Du är en hjälpsam assistent som kan svara på frågor om företagets kollektivavtal.

Svara på frågan baserat på den angivna kontexten, men INTE inkludera dokumentets ID i ditt svar.

Fråga: {question}

Tidigare konversation:
{chat_history}

Kontext:
{context}

Svar:
"""

prompt = PromptTemplate.from_template(template)

# Function to format chat history
def format_history(message_history):
    formatted = ""
    for msg in message_history:
        formatted += f"User: {msg['question']}\nAssistant: {msg['answer']}\n\n"
    return formatted

# Model
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.6,
    api_key=GROQ_API_KEY
)

# Define the chain
message_history = []

chain = {
    "context": retriever,
    "question": RunnablePassthrough(),
    "chat_history": RunnableLambda(lambda x: format_history(message_history))
} | prompt | llm | StrOutputParser()

# Telegram message handler
@dp.message()
async def handle_message(message: Message):
    user_question = message.text

    if user_question.lower() in ["quit", "exit", "sluta", "avsluta"]:
        await message.answer("Tack för idag! Hejdå! 👋")
        return
    
    # Get chatbot response
    result = chain.invoke(user_question)

    # Store conversation history
    message_history.append({
        "question": user_question,
        "answer": result
    })

    await message.answer(result)

# Run the bot
async def main():
    print("Bot is running...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
