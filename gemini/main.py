import os
from dotenv import load_dotenv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import numpy as np
from flask import Flask, request, jsonify

load_dotenv()
test = os.getenv("TEST")
print(test)

# ************************************************************ #

import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


# ************************************************************ #

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# ************************************************************ #


app = Flask(__name__)


@app.route("/", methods=["POST"])
def index():
    try:
        req_data = request.json
        message = req_data.get("message")
        response = model.generate_content(message)
        return f"{response.text}"
    except Exception as e:
        return make_response("Server error", 500)


if __name__ == "__main__":
    app.run(debug=True)

# ************************************************************ #
