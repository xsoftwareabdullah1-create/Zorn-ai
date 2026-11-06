# Serves your Zorn 1.1 model with an OpenAI-style endpoint, so server.js works unchanged.
# pip install flask transformers torch accelerate
# python zorn_serve.py   then:  LLM_BASE_URL=http://localhost:8000/v1  LLM_MODEL=zorn-1.1  node server.js
import os, re, torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = ("You are Zorn 1.1, a smart AI assistant and coding agent. Always reply in English. "
          "First think step by step inside <think></think>, then give the final answer.")
SHOW_THINK = os.environ.get("SHOW_THINK") == "1"

tok = AutoTokenizer.from_pretrained("zorn-1.1")
model = AutoModelForCausalLM.from_pretrained("zorn-1.1", torch_dtype="auto", device_map="auto")
app = Flask(__name__)

@app.post("/v1/chat/completions")
def chat():
    body = request.get_json()
    msgs = [{"role": "system", "content": SYSTEM}] + [m for m in body["messages"] if m["role"] != "system"]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**ids, max_new_tokens=int(body.get("max_tokens", 1024)), do_sample=True,
                         temperature=float(body.get("temperature", 0.4)), top_p=0.9)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    if not SHOW_THINK:
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
    return jsonify({"choices": [{"message": {"role": "assistant", "content": text.strip()}}]})

app.run(port=8000)
