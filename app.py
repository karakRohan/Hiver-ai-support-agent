import json
import pandas as pd
import streamlit as st

from src.agent import run_agent


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AppleSupport AI Agent",
    page_icon="🍎",
    layout="wide",
)


# --------------------------------------------------
# Load Historical Conversations
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed/conversations.csv"
    )


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🍎 AppleSupport AI Customer Support Agent")

st.markdown(
    "AI-powered customer support using historical "
    "AppleSupport conversations."
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("About")

    st.write(
        "This demo classifies customer intent, retrieves "
        "historical support evidence, generates a grounded "
        "reply, and decides whether to auto-handle or escalate."
    )

    st.info(
        "Brand: AppleSupport\n\n"
        "Retrieval: TF-IDF\n\n"
        "Generation: Groq\n\n"
        "Routing: Auto-handle / Escalate"
    )


# --------------------------------------------------
# Customer Input
# --------------------------------------------------

st.subheader("Customer Message")

message = st.text_area(
    "Enter a customer support message:",
    placeholder="Example: My iPhone battery is draining very quickly",
    height=120,
)


# --------------------------------------------------
# Run Agent
# --------------------------------------------------

if st.button("🚀 Analyze Message", type="primary"):

    if not message.strip():
        st.warning("Please enter a customer message.")
        st.stop()

    with st.spinner("Analyzing customer message..."):

        try:
            history = load_data()

            result = run_agent(
                message.strip(),
                history,
            )

        except Exception as exc:
            st.error(f"Agent error: {exc}")
            st.stop()


    # --------------------------------------------------
    # Result Summary
    # --------------------------------------------------

    st.divider()

    st.subheader("Agent Decision")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Intent",
            result["intent"],
        )

    with col2:
        st.metric(
            "Confidence",
            f"{result['confidence']:.0%}",
        )

    with col3:
        action = result["action"].replace(
            "_",
            " ",
        ).upper()

        st.metric(
            "Action",
            action,
        )


    # --------------------------------------------------
    # Reason
    # --------------------------------------------------

    st.subheader("Routing Reason")

    st.info(result["reason"])


    # --------------------------------------------------
    # Generated Reply
    # --------------------------------------------------

    st.subheader("💬 Generated Support Reply")

    st.success(result["reply"])


    # --------------------------------------------------
    # Historical Evidence
    # --------------------------------------------------

    st.subheader("📚 Historical Evidence")

    evidence = result.get(
        "evidence",
        [],
    )

    if evidence:

        for i, item in enumerate(
            evidence,
            start=1,
        ):

            with st.expander(
                f"Case {i} — Similarity: {item['similarity']:.3f}"
            ):

                st.markdown(
                    "**Customer:**"
                )

                st.write(
                    item["customer_text"]
                )

                st.markdown(
                    "**Historical AppleSupport Reply:**"
                )

                st.write(
                    item["historical_reply"]
                )

    else:

        st.warning(
            "No historical evidence was retrieved."
        )