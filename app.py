
import os
import streamlit as st
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from google.colab import userdata
import tempfile
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json # Import json to load job data

# --- Load and Initialize ---
# In a real application, you would load your data from a file.
# For this example, we assume the JSON file is available in the same directory.
try:
    # Load job data from JSON file
    with open('Data/linkedin_jobs.json', 'r') as f:
        jobs_list = json.load(f)
except FileNotFoundError:
    st.error("Job data file (Data/linkedin_jobs.json) not found. Please ensure it exists.")
    jobs_list = [] # Initialize as empty to prevent further errors


@st.cache_resource
def initialize_vector_store_for_streamlit(job_data):
    """Initializes a Chroma vector store and retriever with job data."""
    if not job_data:
        st.error("Job data is empty. Please ensure the data loading cell was run successfully.")
        return None

    documents = [Document(page_content=f"{job['title']} at {job['company']} in {job['location']}", metadata=job) for job in job_data]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunk_docs = []
    for doc in documents:
        for chunk in splitter.split_text(doc.page_content):
            chunk_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    google_api_key = userdata.get('GOOGLE_API_KEY')
    if not google_api_key:
        st.error("Google API Key not found in Colab secrets. Please add it as 'GOOGLE_API_KEY'.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    vectordb = Chroma.from_documents(documents=chunk_docs, embedding=embeddings)

    retriever = vectordb.as_retriever()

    st.success("Vector store and retriever initialized successfully!")
    return retriever


retriever = initialize_vector_store_for_streamlit(jobs_list)


if retriever:
    google_api_key = userdata.get('GOOGLE_API_KEY') # Access secrets in Colab
    if google_api_key:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=google_api_key)
    else:
        st.error("Google API Key not found for LLM. Please add it as 'GOOGLE_API_KEY' in Colab secrets.")
        llm = None

    if llm:
        prompt_template = """
        You are an AI job recommendation assistant.
        Given the user's skills and preferences, recommend the most relevant job openings from the retrieved postings.

        User profile:
        {query}

        Relevant jobs:
        {context}

        Provide recommendations in bullet points with job title, company, and location.
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "context"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
            input_key="query"
        )

        # Streamlit UI
        st.title("💼 AI Job Recommendation System")
        st.write("Discover jobs that fit your skills & preferences.")

        user_input = st.text_area("Enter your skills, experience, and job preferences:")
        # Upload functionality is not fully implemented for resume parsing in this example
        uploaded_file = st.file_uploader("Upload your resume (Optional):", type=["pdf", "docx", "txt"])

        resume_content = ""
        if uploaded_file is not None:
            # In a real application, you would add resume parsing logic here
            st.info(f"Resume '{uploaded_file.name}' uploaded. (Resume parsing functionality not implemented in this example)")
            # Example of how you might read a text file:
            # if uploaded_file.type == "text/plain":
            #     resume_content = uploaded_file.getvalue().decode("utf-8")


        if st.button("Get Recommendations"):
            if user_input.strip() or uploaded_file is not None:
                query = user_input.strip()
                # If only a resume is uploaded, you might process it and use its content as the query
                # For now, we'll just use the user_input text.
                # If you implement resume parsing, you would combine or replace query with resume_content
                if uploaded_file is not None and not query:
                     # This part would be more sophisticated with actual resume parsing
                     st.warning("Resume uploaded but no text input. Resume parsing not implemented.")
                     query = "Recommend jobs based on general data analysis skills." # Fallback or process resume content

                if not query:
                    st.warning("Please enter your profile information or ensure resume processing is added.")
                else:
                    with st.spinner("Finding best matches..."):
                        response = qa_chain.invoke({"query": query})
                    st.subheader("Recommended Jobs")
                    st.write(response["result"])

                    with st.expander("See Source Job Descriptions"):
                        if response["source_documents"]:
                            for i, doc in enumerate(response["source_documents"], start=1):
                                st.markdown(f"**Source {i}:**")
                                st.write(doc.page_content)
                        else:
                            st.info("No source documents found for this query.")
            else:
                st.warning("Please enter your profile information or upload a resume.")

    else:
        st.error("LLM could not be initialized. Please check your Google API Key.")
else:
    st.error("Retriever could not be initialized. Please check the job data and API key.")
