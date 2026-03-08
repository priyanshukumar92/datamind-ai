"""
AI Data Analyst Agent - Core NL to Code Engine
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

# ── API Key ──
api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=api_key)

# ── Blocked keywords for security ──
BLOCKED_KEYWORDS = [
    "os.system", "subprocess", "shutil.rmtree", "shutil.rmdir",
    "open(", "__import__", "eval(", "exec(", "socket",
    "requests", "urllib", "http", "ftplib", "smtplib"
]

def is_code_safe(code: str) -> tuple[bool, str]:
    for keyword in BLOCKED_KEYWORDS:
        if keyword in code:
            return False, f"Blocked keyword detected: '{keyword}'"
    return True, ""

def execute_code(code: str, df: pd.DataFrame) -> dict:
    safe, reason = is_code_safe(code)
    if not safe:
        return {"error": reason, "output": "", "result": None, "chart_data": None}

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    local_vars = {
        "df": df,
        "pd": pd,
    }

    try:
        exec(code, {}, local_vars)
        output = sys.stdout.getvalue()
        result = local_vars.get("result", None)
        chart_data = local_vars.get("chart_data", None)
        return {
            "result": result,
            "output": output,
            "chart_data": chart_data,
            "error": None
        }
    except Exception:
        output = sys.stdout.getvalue()
        error = traceback.format_exc()
        return {
            "result": None,
            "output": output,
            "chart_data": None,
            "error": error
        }
    finally:
        sys.stdout = old_stdout

def extract_code(text: str) -> str:
    pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()

class DataAnalystAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.history = []
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        df = self.df
        col_info = "\n".join([f"  - {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
        sample = df.head(3).to_string()
        stats = df.describe(include="all").to_string()

        return f"""You are an expert Python data analyst. You have access to a pandas DataFrame called `df`.

DATASET INFO:
Columns and types:
{col_info}

Sample rows:
{sample}

Statistics:
{stats}

RULES:
1. Always write Python code inside ```python ``` blocks.
2. Store the final result in a variable called `result`.
3. For charts, store a dict in `chart_data` with keys: type (bar/line/pie/scatter), title, x (list), y (list).
4. Add a brief plain-English explanation before the code block.
5. Never use os, subprocess, socket, requests, or file operations.
6. Keep code simple and correct.
7. When converting columns to numeric, ALWAYS use pd.to_numeric(df['col'], errors='coerce') to handle non-numeric values like "C".
8. After conversion, always drop NaN rows using .dropna() before plotting or aggregating.
"""

    def ask(self, question: str) -> dict:
        self.history.append({"role": "user", "content": question})

        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=1000,
                messages=messages
            )
            reply = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": reply})

            code = extract_code(reply)
            explanation = reply.split("```")[0].strip() if "```" in reply else ""

            exec_result = execute_code(code, self.df)

            # Auto retry on error
            if exec_result["error"] and "```" in reply:
                retry_msg = f"That code had an error:\n{exec_result['error']}\nPlease fix and rewrite the code."
                self.history.append({"role": "user", "content": retry_msg})
                retry_messages = [{"role": "system", "content": self.system_prompt}] + self.history
                retry_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=1000,
                    messages=retry_messages
                )
                retry_reply = retry_response.choices[0].message.content
                self.history.append({"role": "assistant", "content": retry_reply})
                code = extract_code(retry_reply)
                explanation = retry_reply.split("```")[0].strip() if "```" in retry_reply else explanation
                exec_result = execute_code(code, self.df)

            return {
                "explanation": explanation,
                "code": code,
                "result": exec_result.get("result"),
                "output": exec_result.get("output"),
                "chart_data": exec_result.get("chart_data"),
                "error": exec_result.get("error")
            }

        except Exception as e:
            return {
                "explanation": "",
                "code": "",
                "result": None,
                "output": "",
                "chart_data": None,
                "error": str(e)
            }

    def reset(self):
        self.history = []


def ask_about_image(image_file, question: str) -> str:
    try:
        image_data = image_file.read()
        b64 = base64.b64encode(image_data).decode("utf-8")

        filename = getattr(image_file, "name", "image.jpg").lower()
        if filename.endswith(".png"):
            media_type = "image/png"
        elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
            media_type = "image/jpeg"
        else:
            media_type = "image/jpeg"

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Vision error: {str(e)}"