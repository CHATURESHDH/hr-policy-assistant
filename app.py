# ===== IMPORTS =====
import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from typing import TypedDict
from langgraph.graph import StateGraph, END

# ===== SETUP (runs once) =====
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Cache the embedding model so it doesn't reload on every interaction
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Create a fresh in-memory Chroma client each session
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.EphemeralClient()
    st.session_state.collection = None
    st.session_state.chat_history = []
    st.session_state.last_processed_file = None

# ===== DOCUMENT EXTRACTION FUNCTIONS =====

def extract_text_from_pdf(uploaded_file):
    """Extract all text from an uploaded PDF file."""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_txt(uploaded_file):
    """Extract text from an uploaded TXT file."""
    return uploaded_file.read().decode("utf-8")

def dataframe_to_sentences(df):
    """
    Convert each row of a dataframe into a natural-language sentence.
    This makes tabular data much easier for embeddings/semantic search to understand,
    compared to raw column-aligned text.
    """
    sentences = []
    columns = df.columns.tolist()
    for _, row in df.iterrows():
        row_text = ", ".join([f"{col}: {row[col]}" for col in columns])
        sentences.append(row_text)
    return "\n".join(sentences)

def extract_text_from_csv(uploaded_file):
    """Extract text from an uploaded CSV file, converting rows into readable sentences."""
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    return dataframe_to_sentences(df)

def extract_text_from_excel(uploaded_file):
    """Extract text from an uploaded Excel file, converting rows into readable sentences."""
    import pandas as pd
    df = pd.read_excel(uploaded_file)
    return dataframe_to_sentences(df)

def extract_text(uploaded_file):
    """Detect file type and route to the correct extraction function."""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    elif file_name.endswith(".csv"):
        return extract_text_from_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        return extract_text_from_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type")

def process_document(text, doc_name):
    """Chunk text, embed it, and store it in the vector database."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = embedding_model.encode(chunks)

    try:
        st.session_state.chroma_client.delete_collection("hr_docs")
    except Exception:
        pass
    collection = st.session_state.chroma_client.create_collection("hr_docs")

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    )
    st.session_state.collection = collection
    return len(chunks)

# ===== LANGGRAPH AGENT =====
# We model the agent as a graph with 4 nodes:
#   planner -> (router decides RETRIEVE or DIRECT)
#   retrieve_node -> generate_node -> validate_node -> END   (RETRIEVE path)
#   direct_node -> END                                        (DIRECT path)

class AgentState(TypedDict):
    """
    This defines the 'shared memory' that flows through the graph.
    Every node can read from and write to this state.
    """
    question: str
    decision: str
    context: str
    answer: str
    is_valid: bool

def planner_node(state: AgentState) -> AgentState:
    """Node 1: Decide whether the question needs document retrieval or a direct reply."""
    planning_prompt = f"""You are a planning agent. Decide how to handle this user message.

If the message is a greeting, small talk, or doesn't require looking up HR policy
information, respond with exactly: DIRECT

If the message is a genuine question about HR policy, leave, benefits, or company rules,
respond with exactly: RETRIEVE

Message: "{state['question']}"

Respond with ONLY one word: DIRECT or RETRIEVE"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": planning_prompt}]
    )
    decision = response.choices[0].message.content.strip().upper()
    state["decision"] = decision
    return state

def route_after_planner(state: AgentState) -> str:
    """
    This is a 'conditional edge' - it looks at the state and decides
    which node to go to next: 'retrieve' or 'direct'.
    """
    if state["decision"] == "DIRECT":
        return "direct"
    return "retrieve"

def retrieve_node(state: AgentState) -> AgentState:
    """Node 2 (RETRIEVE path): Search the vector store for relevant chunks."""
    query_embedding = embedding_model.encode([state["question"]])
    results = st.session_state.collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )
    retrieved_chunks = results['documents'][0]
    state["context"] = "\n\n".join(retrieved_chunks)
    return state

def generate_node(state: AgentState) -> AgentState:
    """Node 3 (RETRIEVE path): Generate a grounded answer using the retrieved context."""
    prompt = f"""You are an HR policy assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "This isn't covered in the provided policy documents."

Context:
{state['context']}

Question: {state['question']}

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    state["answer"] = response.choices[0].message.content
    return state

def validate_node(state: AgentState) -> AgentState:
    """Node 4 (RETRIEVE path): Check the answer is actually supported by the context."""
    validation_prompt = f"""You are a fact-checker. Determine if the ANSWER below is
reasonably supported by the CONTEXT (it doesn't need to be word-for-word, just
factually consistent). Reply with only YES or NO.

Context:
{state['context']}

Question: {state['question']}
Answer: {state['answer']}

Is the answer reasonably supported by the context? Reply with only YES or NO."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": validation_prompt}]
    )
    is_valid = "YES" in response.choices[0].message.content.strip().upper()
    state["is_valid"] = is_valid

    if not is_valid:
        state["answer"] = "I found some related information, but I'm not confident it fully answers your question. Please verify with HR directly."

    return state

def direct_node(state: AgentState) -> AgentState:
    """Node (DIRECT path): Handle greetings/small talk without document lookup."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a friendly HR assistant chatbot."},
            {"role": "user", "content": state["question"]}
        ]
    )
    state["answer"] = response.choices[0].message.content
    return state

def build_agent_graph():
    """
    Build the LangGraph StateGraph wiring all nodes together:

                        ┌─────────┐
                        │ planner │
                        └────┬────┘
                             │ (conditional route)
                 ┌───────────┴───────────┐
                 ▼                       ▼
           ┌──────────┐             ┌────────┐
           │ retrieve │             │ direct │
           └────┬─────┘             └───┬────┘
                ▼                       │
           ┌──────────┐                 │
           │ generate │                 │
           └────┬─────┘                 │
                ▼                       │
           ┌──────────┐                 │
           │ validate │                 │
           └────┬─────┘                 │
                ▼                       ▼
               END                     END
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("direct", direct_node)

    graph.set_entry_point("planner")

    # Conditional branching after the planner decides
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"retrieve": "retrieve", "direct": "direct"}
    )

    # RETRIEVE path: retrieve -> generate -> validate -> END
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", END)

    # DIRECT path: direct -> END
    graph.add_edge("direct", END)

    return graph.compile()

# Compile the graph once (cached so it isn't rebuilt on every rerun)
@st.cache_resource
def get_agent():
    return build_agent_graph()

agent = get_agent()

def smart_assistant(question):
    """Run the LangGraph agent on a question and return the final answer."""
    initial_state = {
        "question": question,
        "decision": "",
        "context": "",
        "answer": "",
        "is_valid": False
    }
    final_state = agent.invoke(initial_state)
    return final_state["answer"]

# ===== STREAMLIT UI =====

st.set_page_config(page_title="HR Policy Assistant", page_icon="📄")
st.title("📄 HR Policy Assistant")
st.caption("Upload an HR policy document and ask questions about it.")

# --- File upload section ---
uploaded_file = st.file_uploader(
    "Upload a policy document",
    type=["pdf", "txt", "csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_processed_file:
        with st.spinner("Processing document..."):
            text = extract_text(uploaded_file)
            num_chunks = process_document(text, uploaded_file.name)
        st.session_state.last_processed_file = uploaded_file.name
        st.session_state.chat_history = []
        st.success(f"Document processed into {num_chunks} chunks. You can now ask questions.")
    else:
        st.info("Document loaded. Ask a question below, or upload a new document to replace it.")

# --- Chat section ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about the policy...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if st.session_state.collection is None:
            answer = "Please upload a policy document first."
        else:
            with st.spinner("Thinking..."):
                answer = smart_assistant(question)
        st.write(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})