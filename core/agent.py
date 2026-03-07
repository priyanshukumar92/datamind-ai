"""
AI Data Analyst Agent - Core NL → Code Engine
"""

import pandas as pd
import traceback
import io
import sys
import re
import os
import base64
import streamlit as st
from groq import Groq

# ── API Key (works both locally and on Streamlit Cloud) ──
api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=api_key)

BLOCKED_KEYWORDS = [
    "os.system", "subprocess", "shutil.rmtree", "__import__",
    "open(", "eval(", "exec(", "importlib", "socket"
]

def is_safe_code(code: str) -> tuple[bool, str]:
    for keyword in BLOCKED_KEYWORDS:
        if keyword in code:
            return False, f"Blocked keyword detected: `{keyword}`"
    return True, "ok"


def execute_code(code: str, df: pd.DataFrame) -> dict:
    safe, reason = is_safe_code(code)
    if not safe:
        return {"error": reason, "output": "", "result": None, "chart_data": None}

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    local_vars = {
        "df": df.copy(),
        "pd": pd,
        "result": None,
        "chart_data": None,
    }

    try:
        exec(code, {"__builtins__": __builtins__}, local_vars)
        output = buffer.getvalue()
        return {
            "error": None,
            "output": output,
            "result": local_vars.get("result"),
            "chart_data": local_vars.get("chart_data"),
        }
    except Exception:
        return {
            "error": traceback.format_exc(),
            "output": buffer.getvalue(),
            "result": None,
            "chart_data": None,
        }
    finally:
        sys.stdout = old_stdout


def summarize_dataset(df: pd.DataFrame) -> str:
    buf = []
    buf.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    buf.append(f"Columns: {', '.join(df.columns.tolist())}")
    buf.append("\nData types:")
    for col, dtype in df.dtypes.items():
        buf.append(f"  {col}: {dtype}")
    buf.append("\nFirst 3 rows (sample):")
    buf.append(df.head(3).to_string(index=False))
    buf.append("\nBasic stats (numeric columns):")
    try:
        buf.append(df.describe().to_string())
    except Exception:
        buf.append("  (stats unavailable)")
    return "\n".join(buf)


def build_system_prompt(dataset_summary: str) -> str:
    return f"""You are an expert AI data analyst.
Your job is to answer the user's questions about their dataset by writing clean, correct Python/pandas code.

DATASET INFO:
{dataset_summary}

RULES:
1. Always assign the final answer to a variable called `result`.
2. If the answer is a DataFrame or Series, assign it to `result`.
3. If the answer is a number or string, assign it to `result`.
4. If a chart would help, create a dict called `chart_data` with keys:
   - "type": "bar" | "line" | "pie" | "scatter"
   - "x": list of x values
   - "y": list of y values
   - "title": chart title
5. Use print() to explain intermediate steps if helpful.
6. Do NOT use matplotlib or plotly - only produce chart_data dict.
7. Do NOT use file I/O, os, subprocess, or any system calls.
8. The DataFrame is already loaded as `df`. Do not reload it.
9. Wrap your code in a ```python block.

After the code block, write a short plain-English explanation.
"""


def extract_code(llm_response: str) -> str:
    pattern = r"```python\s*(.*?)```"
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_explanation(llm_response: str) -> str:
    cleaned = re.sub(r"```python.*?```", "", llm_response, flags=re.DOTALL).strip()
    return cleaned


class DataAnalystAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.dataset_summary = summarize_dataset(df)
        self.conversation_history = []
        self.system_prompt = build_system_prompt(self.dataset_summary)

    def ask(self, user_question: str) -> dict:
        self.conversation_history.append({
            "role": "user",
            "content": user_question
        })

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=2048,
            messages=[{"role": "system", "content": self.system_prompt}] + self.conversation_history
        )
        llm_reply = response.choices[0].message.content

        self.conversation_history.append({
            "role": "assistant",
            "content": llm_reply
        })

        code = extract_code(llm_reply)
        explanation = extract_explanation(llm_reply)

        exec_result = {}
        if code:
            exec_result = execute_code(code, self.df)

            if exec_result.get("error"):
                fix_message = (
                    f"Your code produced this error:\n{exec_result['error']}\n"
                    f"Please fix the code and try again."
                )
                self.conversation_history.append({"role": "user", "content": fix_message})
                retry_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=2048,
                    messages=[{"role": "system", "content": self.system_prompt}] + self.conversation_history
                )
                retry_reply = retry_response.choices[0].message.content
                self.conversation_history.append({"role": "assistant", "content": retry_reply})
                code = extract_code(retry_reply)
                explanation = extract_explanation(retry_reply)
                exec_result = execute_code(code, self.df) if code else exec_result

        return {
            "question": user_question,
            "code": code,
            "explanation": explanation,
            "result": exec_result.get("result"),
            "output": exec_result.get("output", ""),
            "chart_data": exec_result.get("chart_data"),
            "error": exec_result.get("error"),
        }

    def reset(self):
        self.conversation_history = []
        import base64

def ask_about_image(image_file, question: str) -> str:
    """Send image + question directly to Groq vision model."""
    try:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Vision error: {e}]"