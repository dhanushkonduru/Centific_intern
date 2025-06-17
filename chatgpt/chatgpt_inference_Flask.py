from flask import Flask, request, jsonify, render_template_string
import openai
import re

# Initialize OpenAI client
client = openai.OpenAI(api_key="")  # Replace with your API key

app = Flask(__name__)

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenAI Inference API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        textarea { width: 100%%; padding: 8px; font-size: 14px; }
        button { padding: 10px 20px; font-size: 16px; margin-top: 10px; }
        .response { background: #f4f4f4; padding: 20px; border-radius: 8px; margin-top: 20px; }
        ol { padding-left: 20px; }
    </style>
</head>
<body>
    <h2>Ask a Question</h2>
    <form method="post" action="/inference">
        <label>Prompt:</label><br>
        <textarea name="prompt" rows="2">{{ prompt }}</textarea><br><br>
        <label>Question:</label><br>
        <textarea name="question" rows="2">{{ question }}</textarea><br><br>
        <button type="submit">Submit</button>
    </form>
    {% if response %}
        <div class="response">
            <h3>Response:</h3>
            {{ response|safe }}
        </div>
    {% endif %}
</body>
</html>
"""

def format_response_to_html(text):
    # Convert numbered items into <ol><li>
    items = re.findall(r"\d+\.\s+(.*?)(?=\d+\.|\Z)", text, re.DOTALL)
    if items:
        formatted = "<ol>" + "".join(f"<li>{item.strip()}</li>" for item in items) + "</ol>"
    else:
        formatted = "<p>" + text.replace("\n", "<br>") + "</p>"
    return formatted

@app.route("/")
def home():
    return "Flask OpenAI Inference API is running!"

@app.route("/inference", methods=["GET", "POST"])
def inference():
    prompt = ""
    question = ""
    response_html = None

    if request.method == "POST":
        if request.content_type.startswith("application/x-www-form-urlencoded"):
            prompt = request.form.get("prompt", "")
            question = request.form.get("question", "")
        else:
            data = request.get_json()
            prompt = data.get("prompt", "") if data else ""
            question = data.get("question", "") if data else ""

        if not prompt or not question:
            return jsonify({"error": "Missing prompt or question"}), 400

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question}
                ]
            )
            response_text = response.choices[0].message.content
            print("OpenAI Response:", response_text)

            # Format for HTML display
            response_html = format_response_to_html(response_text)

            # If JSON (like Postman), return JSON
            if request.is_json:
                return jsonify({"response": response_text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template_string(HTML_FORM, prompt=prompt, question=question, response=response_html)

if __name__ == "__main__":
    app.run(debug=True)
