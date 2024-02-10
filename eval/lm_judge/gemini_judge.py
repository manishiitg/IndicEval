import google.generativeai as genai
import re
#### very basic code to test it out!

#https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl
prompt = """
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. 
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. 
Begin your evaluation by providing a short explanation. Be as objective as possible. 

After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "Rating": <rating>, "Explanation": <explanation_for_rating>".
Provide rating for every factor like helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.

Final provide a final rating in format Overall Rating: <overall_rating>

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]
"""

genai.configure(api_key="AIzaSyCsXdyGjzyvLJRBJqjhmXiRV4Gs1GL_edk")
model = genai.GenerativeModel('gemini-pro')

rating_pattern = r'Overall Rating: (\d+(?:\.\d+)?)/'

def get_lm_judge_rating(question, answer):
    prompt_1 = prompt.replace("{question}", question)
    prompt_1 = prompt_1.replace("{answer}", answer)
    response = model.generate_content(prompt_1, generation_config=genai.types.GenerationConfig(temperature=0.0))
    match = re.search(rating_pattern, response.text)

    # If a match is found, extract the rating
    if match:
        rating = match.group(1)
        return response.text, rating
    else:
        raise ValueError("Rating not found")

