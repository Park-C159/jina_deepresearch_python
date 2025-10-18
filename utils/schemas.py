from pydantic import BaseModel, Field

from utils.get_log import get_logger
from utils.safe_generator import ObjectGeneratorSafe

MAX_QUERIES_PER_STEP = 5

log = get_logger("schemas")


def getLanguagePrompt(question: str):
    return {
        "system": """Identifies both the language used and the overall vibe of the question

<rules>
Combine both language and emotional vibe in a descriptive phrase, considering:
  - Language: The primary language or mix of languages used
  - Emotional tone: panic, excitement, frustration, curiosity, etc.
  - Formality level: academic, casual, professional, etc.
  - Domain context: technical, academic, social, etc.
</rules>

<examples>
Question: "fam PLEASE help me calculate the eigenvalues of this 4x4 matrix ASAP!! [matrix details] got an exam tmrw 😭"
Evaluation: {
    "langCode": "en",
    "langStyle": "panicked student English with math jargon"
}

Question: "Can someone explain how tf did Ferrari mess up their pit stop strategy AGAIN?! 🤦‍♂️ #MonacoGP"
Evaluation: {
    "langCode": "en",
    "languageStyle": "frustrated fan English with F1 terminology"
}

Question: "肖老师您好，请您介绍一下最近量子计算领域的三个重大突破，特别是它们在密码学领域的应用价值吗？🤔"
Evaluation: {
    "langCode": "zh",
    "languageStyle": "formal technical Chinese with academic undertones"
}

Question: "Bruder krass, kannst du mir erklären warum meine neural network training loss komplett durchdreht? Hab schon alles probiert 😤"
Evaluation: {
    "langCode": "de",
    "languageStyle": "frustrated German-English tech slang"
}

Question: "Does anyone have insights into the sociopolitical implications of GPT-4's emergence in the Global South, particularly regarding indigenous knowledge systems and linguistic diversity? Looking for a nuanced analysis."
Evaluation: {
    "langCode": "en",
    "languageStyle": "formal academic English with sociological terminology"
}

Question: "what's 7 * 9? need to check something real quick"
Evaluation: {
    "langCode": "en",
    "languageStyle": "casual English"
}
</examples>""",
        "user": question
    }


languageISO6391Map = {
    'en': 'English',
    'zh': 'Chinese',
    'zh-CN': 'Simplified Chinese',
    'zh-TW': 'Traditional Chinese',
    'de': 'German',
    'fr': 'French',
    'es': 'Spanish',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'tr': 'Turkish',
    'nl': 'Dutch',
    'pl': 'Polish',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'da': 'Danish',
    'fi': 'Finnish',
    'el': 'Greek',
    'he': 'Hebrew',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'ms': 'Malay',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'ro': 'Romanian',
    'bg': 'Bulgarian',
}


class LanguageResult(BaseModel):
    lang_code: str = Field(..., max_length=10, description="ISO 639-1 language code")
    lang_style: str = Field(..., max_length=100, description="[vibe & tone] in [what language]")


class Schemas:

    def __init__(self):
        self.languageStyle = 'formal English'
        self.languageCode = 'en'
        self.searchLanguageCode = None

    async def set_language(self, query):
        if languageISO6391Map[query]:
            self.languageCode = query
            self.languageStyle = f'formal ${languageISO6391Map[query]}'
            return

        prompt = getLanguagePrompt(query[:100])
        generator = ObjectGeneratorSafe()

        result = await generator.generate_object({
            "model": 'evaluator',
            "schema": self.get_language_schema(),
            "system": prompt.get("system"),
            "prompt": prompt.get("user"),
        })

        self.languageCode = result.get("lang_code")
        self.languageStyle = result.get("lang_style")
        log.debug(f"language: ${self.languageCode} -> ${self.languageStyle}")

    # ---------------- 各组 schema ----------------
    def get_language_schema(self) -> type[BaseModel]:
        return LanguageResult
