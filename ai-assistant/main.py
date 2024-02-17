import os
from dotenv import load_dotenv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import numpy as np
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS

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
CORS(app)

# app.config["DEBUG"] = False if os.environ.get("PRODUCTION", "False") == "True" else True


@app.route("/", methods=["POST"])
def index():
    try:
        req_data = request.json
        message = req_data.get("message")
        response = model.generate_content(message)
        return f"{response.text}"
    except Exception as e:
        return make_response("Server error", 500)


# ************************************************************ #

palm = genai


support_context = "Imagine you are an AI-powered personal healthcare advisor with the name Alchbot. A user has just reached out to you seeking guidance and support. Write a response that addresses their concerns. If the topic doesn't relate to healthcare or fitness, reply to them by saying that you are unable to help them."


def palmCall(message: str, context: str, history: list):
    valid_history = []
    for i, h in enumerate(history):
        if "content" in h and "author" in h:
            valid_history.append(h)
    valid_history.append({"content": message, "author": "user"})
    reply = palm.chat(context=context, messages=valid_history)
    return reply.last


@app.post("/query")
def query():
    try:
        req_data = request.json
        message = req_data.get("message", "")
        context = req_data.get("context", "")
        history = req_data.get("history", [])
        response = palmCall(message, "" + context, history)
        return f"{response}"
    except Exception as e:
        return make_response("Server error", 500)


@app.post("/support")
def support():
    try:
        req_data = request.json
        message = req_data.get("message")
        context = req_data.get("context", "")
        history = req_data.get("history", [])
        response = palmCall(message, support_context + context, history)
        return f"{response}"
    except Exception as e:
        return make_response("Server error", 500)


# # ************************************************************ #

from googletrans import Translator


@app.post("/translate")
def translate1():
    try:
        translator = Translator()
        req_data = request.json
        message = req_data.get("message", "")
        language = req_data.get("language", "en-IN")

        language = language.split("-")[0]

        response = translator.translate(message, dest=language).text
        return response
    except Exception as e:
        print(e)
        return make_response(f"{e}", 500)


# # ************************************************************ #

# from pydub import AudioSegment
# import subprocess


# @app.post("/convertToSpeech")
# def convertToSpeech():
#     try:
#         audio_file = request.files["audio"]
#         audio_file.save("./audio.mp3")
#         sound = AudioSegment.from_mp3("./audio.mp3")
#         sound.export("./output.ogg", format="ogg")
#         result = subprocess.run(
#             ["rhubarb", "-f", "json", "./output.ogg"], capture_output=True
#         )
#         return f"{result.stdout.decode('utf-8')}"
#     except Exception as e:
#         print(e)
#         return make_response(f"{e}", 500)


# ************************************************************ #

if __name__ == "__main__":
    app.run(debug=True)

# ************************************************************ #
