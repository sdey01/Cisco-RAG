"""Prompts kept separate from orchestration code."""

CLASSIFIER_PROMPT = """Classify the user question into exactly one label: factual or broad.
factual means a direct lookup for a specific fact, command, setting, value, definition, or isolated item.
broad means troubleshooting reasoning, causes, diagnosis, comparison, procedures, or multiple steps.
Return only the lowercase label.

Question: {question}"""

EXPANSION_PROMPT = """Create up to {limit} concise alternative search queries for this Cisco Nexus 9000 troubleshooting question.
Preserve the intent and use relevant technical terminology. Return one query per line and no numbering or commentary.

Question: {question}"""

ANSWER_PROMPT = """Answer the user's question using only the supplied Cisco Nexus 9000 troubleshooting guide excerpts.
Do not invent Cisco commands, settings, causes, or procedures. If the excerpts do not provide enough information, say so clearly.
Give a concise but technically useful answer and cite source/page information inline when possible.

Original user question:
{question}

Retrieved documentation:
{context}"""
