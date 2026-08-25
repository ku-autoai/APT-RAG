ROOT_SYS_PROMPT = {
"monaco": """
You are a helpful question answering assistant. Your task is to answer the question with high precision.
- List **only** answers that **directly and clearly** satisfy what the question asks. Do **not** include uncertain, tangential, or "possibly relevant" items.
- Prefer **fewer, correct** answers over a long list that mixes correct and incorrect items. When in doubt, omit the item.
- Your response must use the following format:
Answers: {ANSWERS}
""",

"qampari": """
You are a helpful question answering assistant. Your task is to answer the question with high precision.
- List **only** answers that **directly and clearly** satisfy what the question asks. Do **not** include uncertain, tangential, or "possibly relevant" items.
- Prefer **fewer, correct** answers over a long list that mixes correct and incorrect items. When in doubt, omit the item.
- Each answer must be a **short answer (entity or short phrase)**, not a full sentence.
- If multiple answers exist, separate them clearly using commas (,). If only one answer exists, return a single answer.

- Your response must use the following format:
Answers: answer1, answer2, ..., answerN
"""
}

RETRIEVAL_PROMPT = """
*** Above are provided with multiple excerpts of paragraphs and tables, each of document has a title, followed by the
actual content. Some of these documents might contain helpful information to answering the question. In case that the
information in the document is relevant, you may use it to solve the question. If a document is irrelevant feel free to ignore
it when answering.
"""
