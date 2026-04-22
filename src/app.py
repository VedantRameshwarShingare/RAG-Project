import streamlit as st
from graph import build_graph

# Build graph
app_graph = build_graph()

st.title("💬 AI Customer Support Assistant (Groq Powered)")

# Input
query = st.text_input("Ask your question:")

# Button
if st.button("Submit"):
    state = {"query": query}

    result = app_graph.invoke(state)

    # Output
    st.write("### Answer")
    st.write(result["answer"])

    st.write("### Confidence")
    st.write(result.get("confidence", 0))

    st.write("### Intent")
    st.write(result.get("intent", "unknown"))

    # HITL Warning
    if result.get("confidence", 0) < 0.5:
        st.warning("⚠️ Escalated to Human Support")