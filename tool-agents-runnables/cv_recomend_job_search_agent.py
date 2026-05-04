
# ============================================================
# 🧠 PROJECT FLOW (MULTI CV AI CAREER AGENT)
# ============================================================
# 1. User selects CV category (Full Stack / Backend / Python)
# 2. User uploads multiple CVs (PDF)
# 3. Each CV is stored separately in Chroma vector DB
# 4. Each CV becomes a "knowledge profile"
# 5. User enters job query
# 6. System selects best matching CV profile
# 7. Relevant CV chunks retrieved via RAG
# 8. Real job search performed (Tavily API)
# 9. CV + Job data combined
# 10. LLM analyzes match
# 11. Suggests improvements
# 12. Generates cover letter
# ============================================================

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from tavily import TavilyClient

st.set_page_config(page_title="Multi CV AI Agent", page_icon="📄")

st.title("Multi CV expert career assistant -  job search agent")

# ============================================================
# 🧠 STEP 1: SELECT CV CATEGORY
# ============================================================

cv_type = st.selectbox("Select CV Type",["Full-Stack","Backend","Frontend","Python"])


# ============================================================
# 📂 STEP 2: UPLOAD MULTIPLE CVs
# ============================================================

uploaded_files = st.file_uploader(
    "Upload CV's",
    type="pdf",
    accept_multiple_files=True 
)

# ============================================================
# 🧠 STEP 3: PROCESS & STORE EACH CV IN SEPARATE VECTOR DB
# ============================================================

if uploaded_files:
    if st.button("Process & Store CVs"):
        for file in uploaded_files:
            # save the file temporary
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                file_path = temp_file.name
            st.info(f"Processing {file.name} as {cv_type}")

            # load PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Split Text
            splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=200)

            chunks = splitter.split_documents(docs)

            # Embeddings
            embeddings = OpenAIEmbeddings()

            # ====================================================
            # 🧠 STEP 4: STORE IN CHROMA (SEPARATE DB PER TYPE)
            # ====================================================

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=f"cv_db/{cv_type.lower()}"
            )

            vectorstore.persist()
        st.success("All CVs stored successfully!")

# ============================================================
# 🔍 STEP 5: LOAD VECTOR DB BASED ON JOB QUERY
# ============================================================

def select_cv_type(job_query:str):
    job_query = job_query.lower()

    if "frontend" in job_query or "react" in job_query:
        return "full stack"
    if "backend" in job_query or "node" in job_query:
        return "backend"
    if "python" in job_query or "ml" in job_query:
        return "python"
    return "full stack"

# ============================================================
# 🧠 STEP 6: JOB ANALYSIS SYSTEM
# ============================================================
st.divider()
st.subheader("🎯 AI Job Matching System")

job_query = st.text_input("Enter job role (e.g. Python Developer, Backend Engineer)")

if job_query:
    # ========================================================
    # 🔎 STEP 7: SELECT BEST CV TYPE
    # ========================================================
    selected_type  =select_cv_type(job_query)

    st.info(f"Selected CV Profile: {selected_type}")

    # ========================================================
    # 📂 STEP 8: LOAD CHROMA VECTOR DB
    # ========================================================
    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        persist_directory=f"cv_db/{selected_type}",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k":4})

    cv_docs = retriever.invoke(job_query)

    cv_text = "\n\n".join([d.page_content for d in cv_docs])

    # ========================================================
    # 🔍 STEP 9: REAL JOB SEARCH (TAVILY)
    # ========================================================

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    job_results = client.search(
        query=f"{job_query} jobs remote and onsite",
        max_results=3 

    )

    jobs = job_results.get("results",[])

    job_text = "\n\n".join([
        f"{j['title']}\n{j['content']}"
        for j in jobs
    ])

    # ========================================================
    # 🧠 STEP 10: LLM SETUP
    # ========================================================

    llm = ChatMistralAI(model="mistral-small-2506")

    # ========================================================
    # ✍️ STEP 11: FINAL PROMPT (ANALYSIS + COVER LETTER)
    # ========================================================

    prompt = ChatPromptTemplate.from_template(
            """
    You are an expert AI career assistant.

    ========================
    📄 CV PROFILE:
    {cv}

    ========================
    💼 JOB LISTINGS:
    {jobs}

    ========================

    TASKS:
    1. Analyze CV vs job match
    2. Tell if candidate is suitable
    3. List missing skills
    4. Suggest improvements
    5. Generate optimized CV summary
    6. Write a professional cover letter

    Be structured and detailed.
    """
    )

    final_prompt = prompt.invoke(
        {
            "cv":cv_text,
            "jobs":job_text
        }
    )

    # ========================================================
    # 🤖 STEP 12: AI RESPONSE
    # ========================================================

    with st.spinner("Analyzing profile..."):

        response = llm.invoke(final_prompt)

    # ========================================================
    # 📊 STEP 13: OUTPUT
    # ========================================================
    st.subheader("🤖 AI Career Report")
    st.write(response.content)







        


