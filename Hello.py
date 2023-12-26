import streamlit as st
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.image(Image.open(Path("./pics/dnas-logo.png")))

st.write("# Welcome to Day & Nite AI Hub! 👋")

st.sidebar.success("☝️ Select a page above.")

st.markdown(
    """
    Day & Nite AI Hub is developed in-house by Leo Guo at Day & Nite.
    The hub hosts a collection of AI bots to empower Day & Nite internal
    staff and its partners.
    **👈 Select a page from the sidebar** to see what Day & Nite AI Hub
    can do for you.
"""
)
st.subheader("Have questions? Want to learn more? Submit feedback?")
st.success("Send an email to Leo Guo at [lguo@wearetheone.com](mailto:lguo@wearetheone.com)", icon="📧")