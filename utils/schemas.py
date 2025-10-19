from typing import List, Type
from pydantic import BaseModel, Field, field_validator

from utils.get_log import get_logger

MAX_QUERIES_PER_STEP = 5
MAX_CLUSTERS = 5
LANGUAGE_STYLE = 'formal English'
LANGUAGE_CODE = 'en'
SEARCH_LANGUAGE_CODE = None
log = get_logger("schemas")


def get_language_prompt():
    return (f'Must in the first-person in "lang:{LANGUAGE_CODE}"; '
            f'in the style of "LANGUAGE_STYLE"')


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


class Cluster(BaseModel):
    question: str = Field(
        ...,
        max_length=100,
        description=(
            'What concrete and specific question this cluster answers. '
            'Should not be general question like "where can I find [what...]"'
        ),
    )
    insight: str = Field(
        ...,
        max_length=200,
        description=(
            'Summary and list key numbers, data, soundbites, and insights that '
            'worth to be highlighted. End with an actionable advice such as '
            '"Visit these URLs if you want to understand [what...]". '
            'Do not use "This cluster..."'
        ),
    )
    urls: List[str]


class ClusterItem(BaseModel):
    question: str = Field(
        ...,
        max_length=100,
        description=(
            'What concrete and specific question this cluster answers. '
            'Should not be general question like "where can I find [what...]"'
        ),
    )
    insight: str = Field(
        ...,
        max_length=200,
        description=(
            'Summary and list key numbers, data, soundbites, and insights that '
            'worth to be highlighted. End with an actionable advice such as '
            '"Visit these URLs if you want to understand [what...]". '
            'Do not use "This cluster..."'
        ),
    )
    urls: List[str]


class SerpClusterPayload(BaseModel):
    think: str = Field(
        max_length=500,
        description=(
            f"Short explain of why you group the search results like this. "
            f"{get_language_prompt()}"
        )
    )
    clusters: List[ClusterItem]
