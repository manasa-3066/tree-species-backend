from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_tree_awareness(tree_name: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert botanist and environmental educator."
                },
                {
                    "role": "user",
                    "content": f"""
                    Explain the tree species "{tree_name}" for public awareness.
                    Include:
                    - Scientific name
                    - Physical characteristics
                    - Medicinal and ecological importance
                    - Environmental benefits
                    - Why this tree should be protected

                    Use simple language for general public.
                    """
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI information currently unavailable. Error: {e}"
