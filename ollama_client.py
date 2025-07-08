
import subprocess

def query_ollama(context, question, model_name="teacher"):
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""
    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8")
