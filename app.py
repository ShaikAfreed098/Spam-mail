import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from flask import Flask, request, render_template_string
import webbrowser
import threading

# ------------------------------
# LOAD DATASETS SEPARATELY
# ------------------------------

email_df = pd.read_csv("email.csv", names=["Category", "Message"], header=None)
spam_df = pd.read_csv("spam.csv")

# ------------------------------
# PREPROCESS DATA
# ------------------------------

email_df = email_df.rename(columns={"Category": "label", "Message": "text"})
spam_df = spam_df.rename(columns={"spam": "label"})
spam_df["label"] = spam_df["label"].apply(lambda x: "spam" if x == 1 else "ham")

# ------------------------------
# TRAIN MODEL
# ------------------------------

X = pd.concat([email_df["text"], spam_df["text"]])
y = pd.concat([email_df["label"], spam_df["label"]])

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression())
])

model.fit(X, y)

# ------------------------------
# FLASK APP WITH AUTO-HIDE RESULT
# ------------------------------

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Email Spam Detector</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

    <style>
        body { background-color: #f8f9fa; }
        .card { margin-top: 60px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        textarea { height: 180px; }
        .result { font-size: 24px; font-weight: bold; }
        .spam { color: red; }
        .ham { color: green; }
    </style>
</head>

<body>
<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-7">
            <div class="card p-4">
                <h3 class="text-center mb-4">📧 Email Spam Detector</h3>

                <form method="POST" id="predictForm">
                    <div class="mb-3">
                        <label class="form-label">Enter Email Text</label>
                        <textarea class="form-control" id="inputText" name="message"
                                  placeholder="Type or paste email or SMS text here...">{{ request.form.get("message","") }}</textarea>
                    </div>

                    <button type="submit" class="btn btn-primary w-100">Predict Spam</button>
                </form>

                {% if result %}
                <div id="predictionBox" class="mt-4 text-center result {% if result=='SPAM' %}spam{% else %}ham{% endif %}">
                    Prediction: {{ result }}
                </div>
                {% else %}
                <div id="predictionBox" style="display:none;"></div>
                {% endif %}
            </div>
        </div>
    </div>
</div>

<script>
    // Hide prediction when input is empty
    const input = document.getElementById("inputText");
    const prediction = document.getElementById("predictionBox");

    function checkInput() {
        if (input.value.trim() === "") {
            prediction.style.display = "none";
        } else if (prediction.innerText.trim() !== "") {
            prediction.style.display = "block";
        }
    }

    input.addEventListener("input", checkInput);

    // Run once when page loads
    checkInput();
</script>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form["message"]
        pred = model.predict([text])[0]
        prediction = "SPAM" if pred == "spam" else "NOT SPAM"

    return render_template_string(HTML_PAGE, result=prediction)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=False)
