import os
import streamlit as st
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from newspaper import Article
from dotenv import load_dotenv

load_dotenv()

# --- State Definition ---
# Define the structure of the state that will be passed between nodes in the graph.
class ResearchState(TypedDict):
    """
    Represents the state of our research agent.

    Attributes:
        topic (str): The research topic provided by the user.
        urls (List[str]): A list of URLs found for the topic.
        articles (List[dict]): A list of dictionaries, each containing a URL and its scraped content.
        summary (str): The final, summarized report in bullet points.
    """
    topic: str
    urls: List[str]
    articles: List[dict]
    summary: str

# --- Tool and Model Initialization ---
# Initialize the tools and models that the agent will use.
try:
    search_tool = TavilySearchResults(max_results=4)
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
except Exception as e:
    st.error(f"Failed to initialize tools. Please check your API keys. Error: {e}")
    st.stop()

# --- Node Functions ---
# These functions represent the individual steps (nodes) in our research workflow.

def search_node(state: ResearchState):
    """
    Searches for relevant articles for the given topic.

    Args:
        state (ResearchState): The current state of the research agent.

    Returns:
        dict: A dictionary containing the list of found URLs.
    """
    st.write("🔍 Searching for relevant articles...")
    results = search_tool.invoke(state['topic'])
    urls = [res['url'] for res in results]
    return {"urls": urls}

def scrape_node(state: ResearchState):
    """
    Scrapes the content from the found URLs.

    Args:
        state (ResearchState): The current state of the research agent.

    Returns:
        dict: A dictionary containing the list of scraped articles.
    """
    st.write("📰 Scraping article content...")
    scraped_articles = []
    for i, url in enumerate(state['urls']):
        try:
            article = Article(url)
            article.download()
            article.parse()
            scraped_articles.append({"url": url, "content": article.text[:4000]}) # Truncate for context window
        except Exception as e:
            st.warning(f"⚠️ Could not scrape {url}: {e}")
    return {"articles": scraped_articles}

def summarize_node(state: ResearchState):
    """
    Generates a combined summary from all scraped articles.

    Args:
        state (ResearchState): The current state of the research agent.

    Returns:
        dict: A dictionary containing the final bullet-point summary.
    """
    st.write("✍️ Generating summary...")
    
    # Create a unified text from all articles for the LLM
    text_content = ""
    for article in state['articles']:
        text_content += f"--- Article from {article['url']} ---\n\n{article['content']}\n\n"

    # Define the prompt for the LLM
    prompt = f"""
    You are an expert research assistant.
    Your task is to create a concise, easy-to-read summary in bullet points based on the provided articles for the topic: "{state['topic']}".

    Instructions:
    1.  Synthesize information from all the articles into a single, cohesive summary.
    2.  Do NOT list summaries for each article separately.
    3.  The summary must be in bullet points.
    4.  The tone should be informative and objective.

    Here is the content from the articles:
    {text_content}
    """
    
    # Invoke the LLM to get the summary
    summary_result = llm.invoke(prompt)
    return {"summary": summary_result.content}

# --- Graph Definition ---
# Define the workflow of the research agent using a StateGraph.

builder = StateGraph(ResearchState)

# Add the nodes to the graph
builder.add_node("search", search_node)
builder.add_node("scrape", scrape_node)
builder.add_node("summarize", summarize_node)

# Define the edges that connect the nodes
builder.add_edge(START, "search")
builder.add_edge("search", "scrape")
builder.add_edge("scrape", "summarize")
builder.add_edge("summarize", END)

# Compile the graph into a runnable object
graph = builder.compile()


# --- Streamlit UI ---
# The user interface for the Research Summarizer app.

st.set_page_config(page_title="Research Summarizer", page_icon="🔎")

st.title("🔎 Research Summarizer")
st.markdown("""
    Enter a topic, and this app will use an AI agent to find, read, and summarize 
    the most relevant articles for you.
""")

topic = st.text_input(
    "Enter your research topic:",
    placeholder="e.g., Learning Styles in High School Students",
    value="" # Set an initial value for demonstration if needed
)

if st.button("Summarize Research"):
    if not topic:
        st.warning("Please enter a research topic.")
    else:
        with st.spinner("The AI agent is working its magic... This may take a moment."):
            try:
                # Run the graph with the initial state
                result = graph.invoke({"topic": topic})
                
                # Display the results
                st.subheader("📝 Summary")
                st.markdown(result['summary'])
                
                st.subheader("📚 Sources")
                for url in result['urls']:
                    st.write(f"- {url}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
