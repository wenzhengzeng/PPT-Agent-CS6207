
from utils import tenacity

REFINE_TEMPLATE_PATH = "prompts/document_refine.txt"

@tenacity
def refine_document(language_model, markdown_document: str):
    """
    Use your refine prompt to convert raw parsed PDF text
    into a structured JSON (doc_json).
    """
    with open(REFINE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        refine_template = f.read()

    prompt = refine_template.replace("{{markdown_document}}", markdown_document)
    doc_json = language_model(prompt, return_json=True)
    if not isinstance(doc_json, dict):
        raise ValueError("Refined document is not in valid JSON format.")
    return doc_json
