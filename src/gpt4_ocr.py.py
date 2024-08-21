import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("GPT4_API_KEY")

def gpt4_ocr(image_text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an OCR engine."},
            {"role": "user", "content": f"Extract text from this receipt image data: {image_text}"}
        ]
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    sample_text = "Image text extracted by image processing"  # Placeholder for actual image-to-text processing
    ocr_text = gpt4_ocr(sample_text)
    print("Extracted OCR Text:", ocr_text)
