import streamlit as st
import pandas as pd
import joblib
import random

# 🔹 Load the trained model
model = joblib.load('bot_model.pkl')

# 🔹 Streamlit Page Config
st.set_page_config(page_title="Smart Bot Detector", layout="centered")
st.title("🛡️ Smart AI Bot Detector")
st.markdown("Detect bots using behavior analysis + CAPTCHA")

# 🔹 Input sliders
mouse_entropy = st.slider("🖱️ Mouse Movement Entropy", 0.0, 1.0, 0.5)
scroll_ratio = st.slider("📜 Scroll Depth Ratio", 0.0, 1.0, 0.8)
time_clicks_avg = st.slider("🕒 Avg Time Between Clicks (sec)", 0.0, 3.0, 1.0)
typing_speed = st.slider("⌨️ Typing Speed (WPM)", 0, 500, 60)
hover_time = st.slider("🕹️ Cursor Hover Time (sec)", 0.0, 2.0, 0.5)
form_fill_ratio = st.slider("📝 Form Fill Percentage", 0.0, 1.0, 0.7)

# 🔹 Make a DataFrame from inputs
input_df = pd.DataFrame([[mouse_entropy, scroll_ratio, time_clicks_avg, typing_speed, hover_time, form_fill_ratio]],
                         columns=["mouse_entropy", "scroll_ratio", "time_clicks_avg", "typing_speed", "hover_time", "form_fill_ratio"])

# 🔹 Predict button
if st.button("🚀 Detect Bot or Human"):
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][int(prediction)]

    # CAPTCHA logic if confidence < 0.7
    if confidence < 0.7:
        captcha = random.randint(1000, 9999)
        st.warning(f"⚠️ Low confidence. Solve CAPTCHA: `{captcha}`")
        user_input = st.text_input("Enter CAPTCHA code below:")
        if user_input == str(captcha):
            st.success("✅ CAPTCHA Passed. You may proceed.")
        else:
            st.error("❌ CAPTCHA Failed. Try again.")
    else:
        if prediction == 0:
            st.error("🛑 Bot Detected (High Confidence)")
        else:
            st.success("✅ Human Detected (High Confidence)")

    # Show bar chart
    st.markdown("### 📊 Your Input Visualization")
    st.bar_chart(input_df.T.rename(columns={0: "Your Input"}))

