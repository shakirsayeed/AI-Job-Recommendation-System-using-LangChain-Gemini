# 💼 AI Job Recommendation System using LangChain & Gemini

This project is an AI-powered job recommendation system that scrapes job postings (e.g., from LinkedIn), stores them in a vector database (Chroma), and recommends the most relevant job opportunities to users based on their skills, experience, and preferences. It uses LangChain for Retrieval-Augmented Generation (RAG) and Google Gemini (Generative AI) for natural language understanding.

## 🚀 Features

🔎 Job Scraping – Scrapes job postings from LinkedIn using Requests + BeautifulSoup or APIs.

📂 Job Storage – Saves job listings in JSON format for reuse.

📊 Vector Embeddings – Converts job data into embeddings using Google Generative AI Embeddings.

📚 Retriever – Uses Chroma vector database to retrieve the most relevant job postings.

🧠 AI-powered Recommendations – Uses Gemini LLM with LangChain’s RetrievalQA to suggest jobs that match user skills.

🌐 Streamlit Web App – Interactive web UI where users can enter their skills/preferences or upload a resume (optional).

📦 Downloadable Project – Includes app.py and requirements.txt for deployment.

## 🧰 Technologies Used

Python 3.10+

Google Colaboratory

Streamlit – Web UI

LangChain – RAG pipeline

ChromaDB – Vector storage for job postings

Google Gemini API – Embeddings + Job Recommendations

BeautifulSoup + Requests – Web scraping

## 🧠 How It Works (Architecture)
1. User Input: You type in your skills and job preferences.
2. Embedding: The app converts your input into numbers (vectors) that the AI can understand.
3. Search: It searches a database of job postings using these vectors to find similar job descriptions.
4. Answer Generation: An AI language model reads the search results and writes job recommendations in easy bullet points.
5. Display: The app shows the recommended jobs on the screen.

## 📊 Data Source
The job postings data is stored locally in a Chroma vector database.
You can add more job postings by updating this database with new data.

## Environment Setup and Variables
Before running the app, you need:

Python installed (version 3.7 or higher recommended)
Install required packages:

    pip install -r requirements.txt


How to Run the App

Open a terminal or command prompt.

Navigate to the project directory.

Run the Streamlit app with:

streamlit run app.py The app will automatically open in your default web browser.

Enter your skills and job preferences in the input box.

Click the Get Recommendations button to see job suggestions.

How It Works The app creates embeddings for job postings using Google's embedding model.

Job data is stored and searched efficiently using Chroma vector database.

When a user inputs their profile, the app converts it into an embedding.

The most relevant job postings are retrieved based on similarity.

A Google Generative AI language model generates readable recommendations from these results.

Example Input Skills: Python, SQL, Excel; Experience: 0 years; Looking for Data Analyst roles in India
