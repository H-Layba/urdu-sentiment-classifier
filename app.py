import gradio as gr
from transformers import pipeline

# Load model from HuggingFace Hub
MODEL_ID = "H-Layba/urdu-sentiment-classifier"

classifier = pipeline(
    "text-classification",
    model=MODEL_ID
)

# Example sentences
EXAMPLES = [
    ["یہ فلم بہت اچھی تھی"],        # This film was very good
    ["آج کا دن بہت برا تھا"],        # Today was a very bad day
    ["کھانا لاجواب تھا"],             # The food was excellent
    ["خدمت بہت خراب تھی"],           # The service was very bad
    ["یہ کتاب بہت مفید ہے"],          # This book is very useful
    ["بہت مایوسی ہوئی"],             # Very disappointed
]

def analyze(text):
    if not text.strip():
        return "", "", 0

    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]

    emoji = "🟢 مثبت (Positive)" if label == "positive" else "🔴 منفی (Negative)"
    confidence = round(score * 100, 2)
    bar = score

    return emoji, f"{confidence}%", bar


# Custom CSS for a clean dark UI with Urdu-friendly typography
css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&family=Inter:wght@400;500;600&display=swap');

body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'Inter', sans-serif;
}

#title {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
}

#title h1 {
    font-size: 2rem;
    font-weight: 600;
    color: #e4e4f0;
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
}

#title p {
    color: #6b6b8a;
    font-size: 0.95rem;
}

.urdu-input textarea {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    font-size: 1.3rem !important;
    direction: rtl !important;
    text-align: right !important;
    line-height: 2.2 !important;
    background: #12121e !important;
    border: 1px solid #2a2a3e !important;
    color: #e4e4f0 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

.urdu-input textarea:focus {
    border-color: #7c6dfa !important;
    box-shadow: 0 0 0 2px rgba(124, 109, 250, 0.15) !important;
}

.result-label {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    text-align: center !important;
    padding: 1rem !important;
    background: #12121e !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 10px !important;
    color: #e4e4f0 !important;
}

.confidence-text {
    font-size: 1.1rem !important;
    text-align: center !important;
    color: #7c6dfa !important;
    font-weight: 500 !important;
    background: #12121e !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 10px !important;
}

button.primary {
    background: #7c6dfa !important;
    border: none !important;
    color: white !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
}

button.primary:hover {
    background: #6b5ce7 !important;
}

.example-btn {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    font-size: 1rem !important;
}

#stats {
    display: flex;
    justify-content: center;
    gap: 2rem;
    padding: 1rem;
    margin-top: 1rem;
}

.stat-card {
    background: #12121e;
    border: 1px solid #2a2a3e;
    border-radius: 10px;
    padding: 0.8rem 1.5rem;
    text-align: center;
}

.stat-num {
    font-size: 1.4rem;
    font-weight: 600;
    color: #7c6dfa;
    font-family: 'JetBrains Mono', monospace;
}

.stat-label {
    font-size: 0.75rem;
    color: #6b6b8a;
    margin-top: 2px;
}

footer { display: none !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:

    gr.HTML("""
        <div id="title">
            <h1>🇵🇰 Urdu Sentiment Analyser</h1>
            <p>اردو متن کا جذباتی تجزیہ — Fine-tuned mBERT on 50,000 Urdu reviews</p>
        </div>
        <div id="stats">
            <div class="stat-card">
                <div class="stat-num">81%</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">0.81</div>
                <div class="stat-label">F1 Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">50K</div>
                <div class="stat-label">Training rows</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">mBERT</div>
                <div class="stat-label">Base model</div>
            </div>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="اردو متن یہاں لکھیں (Enter Urdu text)",
                placeholder="یہاں اردو میں لکھیں...",
                lines=4,
                elem_classes="urdu-input"
            )
            analyze_btn = gr.Button("تجزیہ کریں — Analyze", variant="primary")

        with gr.Column(scale=2):
            label_out = gr.Textbox(
                label="نتیجہ — Result",
                interactive=False,
                elem_classes="result-label"
            )
            confidence_out = gr.Textbox(
                label="اعتماد — Confidence",
                interactive=False,
                elem_classes="confidence-text"
            )
            bar_out = gr.Slider(
                label="Confidence Score",
                minimum=0,
                maximum=1,
                interactive=False
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=text_input,
        label="مثالیں — Try these examples",
        elem_id="examples"
    )

    gr.HTML("""
        <div style="text-align:center; padding: 1.5rem; color: #3a3a5a; font-size: 0.8rem;">
            Built by <a href="https://huggingface.co/H-Layba" style="color:#7c6dfa">H-Layba</a> ·
            Model: <a href="https://huggingface.co/H-Layba/urdu-sentiment-classifier" style="color:#7c6dfa">urdu-sentiment-classifier</a> ·
            Dataset: IMDB Urdu Reviews (50K)
        </div>
    """)

    analyze_btn.click(
        fn=analyze,
        inputs=text_input,
        outputs=[label_out, confidence_out, bar_out]
    )
    text_input.submit(
        fn=analyze,
        inputs=text_input,
        outputs=[label_out, confidence_out, bar_out]
    )

demo.launch()
