import streamlit as st
import os
import requests
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool

from tavily import TavilyClient

# =========================
# 🔐 Load ENV
# =========================
load_dotenv()

# =========================
# 🌦️ Weather Tool
# =========================
def get_weather(city: str) -> str:
    """Get current weather of a city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Missing OpenWeather API key."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    res = requests.get(url).json()

    if str(res.get("cod")) != "200":
        return f"Error: {res.get('message')}"

    temp = res["main"]["temp"]
    desc = res["weather"][0]["description"]

    return f"🌤️ Weather in {city}: {desc}, {temp}°C"


weather_tool = StructuredTool.from_function(get_weather)

# =========================
# 📰 News Tool (Tavily)
# =========================
def get_news(city: str) -> str:
    """Get latest news about a city."""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Missing Tavily API key."

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=f"Latest news in {city}",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}."

    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")

        news_list.append(
            f"**{title}**\n{content[:120]}...\n🔗 {url}"
        )

    return "\n\n".join(news_list)


news_tool = StructuredTool.from_function(get_news)

# =========================
# 🧠 LLM Setup
# =========================
llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.3
)

agent = create_agent(
    model=llm,
    tools=[weather_tool, news_tool],
    system_prompt="You are a helpful city assistant. You can provide weather and news."
)

# =========================
# 🎨 Streamlit UI
# =========================
st.set_page_config(page_title="City Assistant", page_icon="🌍")

st.title("🌍 AI City Assistant")
st.caption("Get Weather 🌦️ and News 📰 instantly")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Ask something like: Weather in Dhaka or News in Chattogram")

if user_input:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent call
    with st.spinner("Thinking..."):
        response = agent.invoke({
            "messages": st.session_state.messages
        })

    bot_reply = response["messages"][-1].content

    # Save bot reply
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    with st.chat_message("assistant"):
        st.markdown(bot_reply)