"""Prompts used by AREA."""

SYSTEM_PROMPT = """\
Answer the question about the given image. Output ONLY the answer, as few words as possible (typically 1-5 words). No explanation, no full sentences.\
"""

SELF_ELICIT_SYSTEM_PROMPT_VQA_TEXT = "\n<START_IMPORTANT_TXT> and <END_IMPORTANT_TXT> are used to mark the important textual evidence. Do not output the markers."
SELF_ELICIT_SYSTEM_PROMPT_VQA_IMG = "\n<START_IMPORTANT_IMG> and <END_IMPORTANT_IMG> are used to mark the important visual evidence. Do not output the markers."

CONTEXT_VQA_PROMPT_training = """\
{question}

The following paragraphs may contain useful information:

{context}

Answer in as few words as possible (1-5 words). Output only the answer.
If the answer is a phrase inside the context, copy only that phrase.
"""

CONTEXT_VQA_PROMPT_SELF_ELICIT = """\
Directly answer the question based on the context passages, no explanation is needed.
If the context does not contain any evidence, output "I cannot answer based on the given context."

Question: {question}

Context:
{context}
"""

VQA_PROMPT = "Answer the question based on the image above.\n\nQuestion: {question}\n\n"
