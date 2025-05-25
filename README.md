# EMI/emiAI Project

## Project Purpose

The EMI/emiAI project is designed as a specialized Question Answering (QA) system tailored for educational contexts, specifically for interacting with school course documents provided in PDF format. Its core purpose is to enable students and potentially educators to query a collection of local PDF documents and receive relevant answers based *only* on the information contained within those documents.

This project leverages the power of Retrieval Augmented Generation (RAG), a technique that combines information retrieval with text generation. Instead of relying solely on a large language model's (LLM) pre-existing knowledge, the RAG approach first retrieves relevant information from a specific knowledge base (in this case, the course PDFs) and then uses an LLM to generate an answer grounded in that retrieved information.

The key requirements and goals of the EMI/emiAI project are:

*   **Document-Specific Answers:** To provide answers that are strictly based on the content of the uploaded course PDFs, preventing the LLM from hallucinating or providing information from outside the defined knowledge base.
*   **Local Knowledge Base:** To build and query a knowledge base derived from local PDF files, ensuring data privacy and allowing the system to be used offline or within a controlled environment.
*   **Efficient Information Retrieval:** To quickly find the most relevant sections within potentially large volumes of text across multiple documents that pertain to a user's query.
*   **Contextual Answer Generation:** To use the retrieved, relevant document snippets as context for a language model to formulate coherent and accurate answers.
*   **Administrator-Managed Content:** The system is designed for administrators (like professors or teaching assistants) to manage the document collection, ensuring that students are querying an authorized and relevant set of materials. It explicitly does *not* allow end-users (students) to upload their own documents, maintaining control over the knowledge base.
*   **User-Friendly Interface:** To provide an intuitive chat interface (using Streamlit) where users can easily ask questions and view responses along with the sources of the information.

In essence, EMI/emiAI acts as an intelligent assistant that makes the information within course PDFs easily accessible and queryable, transforming static documents into a dynamic and interactive knowledge source for students.

## Getting Started

(Further sections on setup, running the app, etc. can be added here)
