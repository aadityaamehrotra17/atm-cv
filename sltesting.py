import streamlit as st

html_content = """
<div style="font-size: 20px; background-color: #f0f0f0; padding: 10px;">
    <div style="padding: 5px; margin-bottom: 5px;">
        <strong>Nish</strong><br>
        Account Balance: $5,000<br>
        Last Transaction: $200<br>
        Membership: Gold
    </div>
</div>
"""
st.markdown(html_content, unsafe_allow_html=True)
