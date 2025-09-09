from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client(api_key="your_api_key")

prompt = (
    """
    I am going to detect edges by filtering red color range.

    highlight every edges of provided site plan image with RED borders.
    """
)

image = Image.open("finalized_plan copy.png")
site_plan_image = Image.open("site plan.png")

response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[image, prompt, "highlight EVERY edges of provided site plan image with RED borders."],
)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save("generated_image5.png")