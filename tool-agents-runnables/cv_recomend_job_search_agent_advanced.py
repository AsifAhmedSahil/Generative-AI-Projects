
# ============================================================
# 🧠 AI CAREER AGENT - ADVANCED SYSTEM
# ============================================================
# FEATURES:
# 1. Multiple CV upload
# 2. Chroma Vector DB (RAG)
# 3. LLM-based CV selection (NO RULES)
# 4. CV scoring system (0-10)
# 5. Job ranking engine
# 6. Chat mode assistant
# 7. Cover letter generation
# ============================================================

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from tavily import TavilyClient

load_dotenv()

st.set_page_config(page_title="AI Career Agent", page_icon="🧠")

st.title("🧠 AI Career Agent (Pro Level)")

# ============================================================
# 📂 STEP 1: UPLOAD MULTIPLE CVs
# ============================================================
uploaded_files = st.file_uploader(
    "Upload CVs (Multiple Allowed)",
    type="pdf",
    accept_multiple_files=True
)

# ============================================================
# 🧠 STEP 2: PROCESS CVs INTO CHROMA
# ============================================================
if uploaded_files:

    if st.button("Process CVs"):

        for file in uploaded_files:

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                path = tmp.name

            loader = PyPDFLoader(path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150
            )

            chunks = splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings()

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="cv_db"
            )

            vectorstore.persist()

        st.success("CVs processed successfully!")

# ============================================================
# 🧠 STEP 3: LOAD VECTOR DB
# ============================================================
if os.path.exists("cv_db"):

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="cv_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatMistralAI(model="mistral-small-2506")

    # ========================================================
    # 💬 STEP 4: CHAT MODE
    # ========================================================
    st.divider()
    st.subheader("💬 Career Chat Mode")

    chat_input = st.chat_input("Ask anything about your career...")

    if chat_input:

        cv_context = retriever.invoke(chat_input)
        context_text = "\n\n".join([d.page_content for d in cv_context])

        chat_prompt = ChatPromptTemplate.from_template(
            """
You are a career AI assistant.

CV DATA:
{cv}

USER QUESTION:
{question}

Answer clearly and give career advice.
"""
        )

        response = llm.invoke(chat_prompt.invoke({
            "cv": context_text,
            "question": chat_input
        }))

        st.write(response.content)

    # ========================================================
    # 🎯 STEP 5: JOB INPUT
    # ========================================================
    st.divider()
    st.subheader("🎯 Job Matching Engine")

    job_query = st.text_input("Enter Job Role")

    if job_query:

        # ====================================================
        # 🧠 STEP 6: LLM CV AUTO SELECTOR (NO RULES)
        # ====================================================
        selector_prompt = ChatPromptTemplate.from_template(
            """
You have multiple CV profiles in database.

Select BEST matching CV for this job:

JOB:
{job}

Explain only which CV type fits best (Full Stack / Backend / Python etc.)
"""
        )

        selection = llm.invoke(selector_prompt.invoke({"job": job_query}))

        selected_cv_type = selection.content

        st.info(f"AI Selected CV: {selected_cv_type}")
# update
        # ====================================================
        # 📄 STEP 7: RETRIEVE CV
        # ====================================================
        docs = retriever.invoke(job_query)
        cv_text = "\n\n".join([d.page_content for d in docs])

        # ====================================================
        # 🔍 STEP 8: JOB SEARCH
        # ====================================================
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        jobs = client.search(
            query=job_query,
            max_results=5
        ).get("results", [])

        job_text = "\n\n".join([
            f"{j['title']}\n{j['content']}"
            for j in jobs
        ])

        # ====================================================
        # 📊 STEP 9: CV SCORING + JOB RANKING
        # ====================================================
        ranking_prompt = ChatPromptTemplate.from_template(
            """
You are an AI recruiter.

CV:
{cv}

Jobs:
{jobs}

TASKS:
1. Score CV from 0 to 10
2. Rank jobs from best to worst match
3. Explain reasoning
4. Suggest improvements
5. Write cover letter for best job
"""
        )

        result = llm.invoke(ranking_prompt.invoke({
            "cv": cv_text,
            "jobs": job_text
        }))

        st.subheader("🤖 AI Career Report")
        st.write(result.content)





