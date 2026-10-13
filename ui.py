import streamlit as st
import requests

# Page Config
st.set_page_config(page_title="Aether Gateway", layout="wide")
st.title("🌌 Aether Gateway: ML Predictive Router")
st.markdown("Dynamic routing across `qwen2.5:0.5b` (Cheap) and `qwen2.5:3b` (Strong) with Semantic Caching.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question to test the ML router..."):
    # Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call FastAPI Backend
    try:
        with st.spinner("Routing and generating..."):
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"message": prompt, "tier": "auto"},
                timeout=120
            )
            data = response.json()
        
        bot_reply = data.get("response", "Error: Invalid response structure.")
        source = data.get("source", "unknown")
        router_tel = data.get("router_telemetry", {})
        llm_tel = data.get("llm_telemetry", {})
        
        # Display Assistant Response & Telemetry
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
            
            # Thesis Metrics Dashboard
            with st.expander("📊 Gateway Telemetry & Thesis Metrics", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Routed Tier", router_tel.get("routed_tier", "N/A").upper())
                    st.metric("Source", "Semantic Cache ⚡" if source == "cache" else "Live LLM 🧠")
                
                with col2:
                    st.metric("Router Latency", f"{router_tel.get('router_latency_ms', 0)} ms")
                    if source == "live_llm":
                        st.metric("LLM Gen Time", f"{llm_tel.get('latency_seconds', 0)} s")
                    else:
                        st.metric("Cache Distance", f"{data.get('distance_metric', 0):.4f}")
                        
                with col3:
                    st.metric("Router Confidence", f"{router_tel.get('confidence', 0)*100:.1f}%")
                    st.caption(f"Raw Logits: {router_tel.get('logits', [])}")

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        
    except requests.exceptions.ConnectionError:
        st.error("🚨 Connection Error: Cannot reach FastAPI backend. Is Uvicorn running on port 8000?")
