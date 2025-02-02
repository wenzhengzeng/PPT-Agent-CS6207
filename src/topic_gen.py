
TOPIC_GENERATE_PROMPT_PATH = "prompts/topic_generate.txt"

def topic_generate(language_model, topic: str):
    """Generate a JSON doc structure from a given text topic using your language model."""
    with open(TOPIC_GENERATE_PROMPT_PATH, "r", encoding="utf-8") as f:
        topic_generate_prompt = f.read()
    prompt = topic_generate_prompt.replace("{{my_topic_placeholder}}", topic)

    text = language_model(prompt, return_json=True)
    if not isinstance(text, dict):
        raise ValueError("Text is not in JSON format or could not parse model output.")
    return text