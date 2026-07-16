import os
import pickle
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# 1. Exact Path Mappings from your Server Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer_pkl.pkl')
model_path = os.path.join(BASE_DIR, 'logistic_pkl (1).pkl')

# Load the vectorizer and model safely using exact filenames
try:
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Success: Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading pickle files: {e}")
    vectorizer = None
    model = None

# 2. Combined HTML/CSS/JS Frontend Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Text Analysis Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col justify-between selection:bg-indigo-500 selection:text-white">

    <!-- Navbar -->
    <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-5xl mx-auto px-6 py-4 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <div class="h-9 w-9 bg-gradient-to-tr from-indigo-500 to-violet-500 rounded-xl flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/20">Ω</div>
                <span class="text-lg font-bold tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">Analytica AI</span>
            </div>
            <span class="text-xs font-medium px-2.5 py-1 bg-indigo-500/10 text-indigo-400 rounded-full border border-indigo-500/20">Logistic Regression</span>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-3xl mx-auto px-6 py-12 w-full flex-grow flex flex-col justify-center">
        <div class="text-center mb-8">
            <h1 class="text-4xl font-extrabold tracking-tight sm:text-5xl bg-gradient-to-r from-white via-slate-200 to-slate-500 bg-clip-text text-transparent">
                Text Sentiment & Analysis
            </h1>
            <p class="mt-3 text-slate-400 max-w-md mx-auto text-base">
                Drop your text or phrases below to evaluate insights instantly powered by natural language processing.
            </p>
        </div>

        <!-- Card Container -->
        <div class="bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 shadow-xl backdrop-blur-xl">
            <form id="analysisForm" class="space-y-4">
                <div>
                    <label for="textInput" class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Insert Content</label>
                    <textarea id="textInput" rows="5" 
                        class="w-full bg-slate-900/60 border border-slate-700 rounded-xl p-4 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition duration-200 resize-none text-sm"
                        placeholder="Type or paste your message here..."></textarea>
                </div>

                <button type="submit" id="submitBtn"
                    class="w-full bg-gradient-to-r from-indigo-500 to-violet-600 hover:from-indigo-600 hover:to-violet-700 text-white font-medium py-3 px-4 rounded-xl transition duration-200 transform active:scale-[0.99] shadow-lg shadow-indigo-500/20 focus:outline-none focus:ring-2 focus:ring-indigo-500/50">
                    Analyze Content
                </button>
            </form>

            <!-- Results Section -->
            <div id="resultContainer" class="hidden mt-6 pt-6 border-t border-slate-700/60">
                <div class="flex items-center justify-between p-4 rounded-xl bg-slate-900/40 border border-slate-700/50">
                    <div>
                        <p class="text-xs font-semibold tracking-wider text-slate-400 uppercase">Analysis Output</p>
                        <h3 id="predictionText" class="text-xl font-bold mt-1"></h3>
                    </div>
                    <div class="text-right">
                        <p class="text-xs font-semibold tracking-wider text-slate-400 uppercase">Confidence</p>
                        <span id="confidenceText" class="text-sm font-mono font-bold text-indigo-400 bg-indigo-500/10 px-2.5 py-1 rounded-md mt-1 inline-block border border-indigo-500/20"></span>
                    </div>
                </div>
            </div>

            <!-- Error Alerts -->
            <div id="errorContainer" class="hidden mt-4 p-4 bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-xl"></div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="text-center py-6 text-xs text-slate-500 border-t border-slate-800/60">
        &copy; 2026 Analytica AI. All rights reserved.
    </footer>

    <!-- Interactive Script -->
    <script>
        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const textInput = document.getElementById('textInput').value;
            const submitBtn = document.getElementById('submitBtn');
            const resultContainer = document.getElementById('resultContainer');
            const errorContainer = document.getElementById('errorContainer');
            const predictionText = document.getElementById('predictionText');
            const confidenceText = document.getElementById('confidenceText');

            errorContainer.classList.add('hidden');
            resultContainer.classList.add('hidden');
            
            if(!textInput.trim()) {
                errorContainer.innerText = "Please input valid characters or words.";
                errorContainer.classList.remove('hidden');
                return;
            }

            submitBtn.disabled = true;
            submitBtn.innerText = "Processing Data...";

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: textInput })
                });

                const result = await response.json();

                if (response.ok) {
                    predictionText.innerText = result.prediction;
                    confidenceText.innerText = result.confidence;
                    
                    if(result.prediction === "Positive") {
                        predictionText.className = "text-xl font-bold mt-1 text-emerald-400";
                    } else {
                        predictionText.className = "text-xl font-bold mt-1 text-rose-400";
                    }
                    
                    resultContainer.classList.remove('hidden');
                } else {
                    errorContainer.innerText = result.error || "An anomaly occurred.";
                    errorContainer.classList.remove('hidden');
                }
            } catch (err) {
                errorContainer.innerText = "Failed to establish communication with the server.";
                errorContainer.classList.remove('hidden');
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerText = "Analyze Content";
            }
        });
    </script>
</body>
</html>
"""

# 3. Routes Configuration
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if vectorizer is None or model is None:
        return jsonify({
            'error': f'Model load mismatch. Server files detected: {os.listdir(BASE_DIR)}.'
        }), 500
    
    data = request.get_json() or {}
    user_text = data.get('text', '').strip()
    
    if not user_text:
        return jsonify({'error': 'Please enter some text to analyze.'}), 400
    
    try:
        # Vectorize and Predict
        vectorized_text = vectorizer.transform([user_text])
        prediction = model.predict(vectorized_text)[0]
        
        # Calculate Confidence Scores
        probabilities = model.predict_proba(vectorized_text)[0]
        confidence = max(probabilities) * 100
        
        # --- DEBUG LOGGING ON RENDER CONSOLE ---
        # Look at your Render Live Logs to see these outputs when you submit text
        print(f"[DEBUG LOG] Input Text: '{user_text}'")
        print(f"[DEBUG LOG] Raw Prediction Value: {prediction}")
        print(f"[DEBUG LOG] Class Probabilities [Class 0, Class 1]: {probabilities}")
        
        # CRITICAL FIX STEP: If your model treats 0 as Positive and 1 as Negative,
        # flip the labels below to perfectly match your training data intent!
        if int(prediction) == 1:
            result_label = "Positive"
        else:
            result_label = "Negative"
        
        return jsonify({
            'prediction': result_label,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
