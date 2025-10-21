from utils.get_log import get_logger
from utils.safe_generator import ObjectGeneratorSafe
from utils.schemas import QuestionEvaluateSchema

TOOL_NAME = 'evaluator'


def get_question_evaluation_prompt(question: str):
    return {
        "system": """You are an evaluator that determines if a question requires definitive, freshness, plurality, and/or completeness checks.

<evaluation_types>
definitive - Checks if the question requires a definitive answer or if uncertainty is acceptable (open-ended, speculative, discussion-based)
freshness - Checks if the question is time-sensitive or requires very recent information
plurality - Checks if the question asks for multiple items, examples, or a specific count or enumeration
completeness - Checks if the question explicitly mentions multiple named elements that all need to be addressed
</evaluation_types>

<rules>
1. Definitive Evaluation:
   - Required for ALMOST ALL questions - assume by default that definitive evaluation is needed
   - Not required ONLY for questions that are genuinely impossible to evaluate definitively
   - Examples of impossible questions: paradoxes, questions beyond all possible knowledge
   - Even subjective-seeming questions can be evaluated definitively based on evidence
   - Future scenarios can be evaluated definitively based on current trends and information
   - Look for cases where the question is inherently unanswerable by any possible means

2. Freshness Evaluation:
   - Required for questions about current state, recent events, or time-sensitive information
   - Required for: prices, versions, leadership positions, status updates
   - Look for terms: "current", "latest", "recent", "now", "today", "new"
   - Consider company positions, product versions, market data time-sensitive

3. Plurality Evaluation:
   - ONLY apply when completeness check is NOT triggered
   - Required when question asks for multiple examples, items, or specific counts
   - Check for: numbers ("5 examples"), list requests ("list the ways"), enumeration requests
   - Look for: "examples", "list", "enumerate", "ways to", "methods for", "several"
   - Focus on requests for QUANTITY of items or examples

4. Completeness Evaluation:
   - Takes precedence over plurality check - if completeness applies, set plurality to false
   - Required when question EXPLICITLY mentions multiple named elements that all need to be addressed
   - This includes:
     * Named aspects or dimensions: "economic, social, and environmental factors"
     * Named entities: "Apple, Microsoft, and Google", "Biden and Trump"
     * Named products: "iPhone 15 and Samsung Galaxy S24"
     * Named locations: "New York, Paris, and Tokyo"
     * Named time periods: "Renaissance and Industrial Revolution"
   - Look for explicitly named elements separated by commas, "and", "or", bullets
   - Example patterns: "comparing X and Y", "differences between A, B, and C", "both P and Q"
   - DO NOT trigger for elements that aren't specifically named
</rules>

<examples>
<example-1>
谁发明了微积分？牛顿和莱布尼兹各自的贡献是什么？
<think>
这是关于微积分历史的问题，不涉及需要最新信息的内容。问题明确提到了牛顿和莱布尼兹两位数学家，要求分析他们各自的贡献，所以需要全面评估这两个特定的方面。这个问题涉及历史事实，有明确的学术研究可以参考，因此需要确定性评估。
</think>
<output>
{"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true}
</output>
</example-1>

<example-2>
fam PLEASE help me calculate the eigenvalues of this 4x4 matrix ASAP!! [matrix details] got an exam tmrw 😭
<think>
This is a mathematical question about eigenvalues which doesn't change over time, so no need for recent information. A 4x4 matrix has multiple eigenvalues, so this requires identifying several distinct values. This is a pure mathematics problem with precise, verifiable solutions that can be definitively evaluated. The question asks for calculation of eigenvalues only, not addressing multiple distinct topics.
</think>
<output>
{"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": true,
"needsCompleteness": false}
</output>
</example-2>

<example-3>
Quelles sont les principales différences entre le romantisme et le réalisme dans la littérature du 19ème siècle?
<think>
C'est une question sur l'histoire littéraire, donc aucun besoin d'informations récentes. La question mentionne spécifiquement deux mouvements: le romantisme et le réalisme. Je dois évaluer ces deux éléments nommés, donc l'exhaustivité est importante ici. Cette question porte sur des concepts littéraires établis avec des caractéristiques documentées, donc une évaluation définitive est possible. La question ne demande pas une liste ou énumération multiple au-delà des deux mouvements spécifiés.
</think>
<output>
{"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true}
</output>
</example-3>

<example-4>
Shakespeare の最も有名な悲劇を5つ挙げ、簡単にあらすじを説明してください。
<think>
シェイクスピアの悲劇についての質問であり、時事的な情報は不要。「5つ挙げ」という指定があるため、複数の項目が求められている。「最も有名な」という基準は学術的コンセンサスや文化的重要性に基づいて判断できるため、確定的な評価が可能。特定の作品を分析するよう求められているわけではなく、複数の作品を列挙することが主な要件。
</think>
<output>
{"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": true,
"needsCompleteness": false}
</output>
</example-4>

<example-5>
What are the current interest rates for mortgage loans from Bank of America, Wells Fargo, and Chase Bank in the US?
<think>
This question asks about "current" interest rates, so it clearly requires up-to-date information. The query specifically names three banks: Bank of America, Wells Fargo, and Chase Bank. Each of these named entities must be addressed, making completeness necessary. This question seeks factual financial data that can be objectively verified, so definitive evaluation is needed. The question isn't asking for multiple types of information beyond the specified banks.
</think>
<output>
{"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": false,
"needsCompleteness": true}
</output>
</example-5>

<example-6>
2025年に注目すべき人工知能の3つのトレンドは何ですか？
<think>
これは将来のAIトレンドに関する質問なので、最新の情報が必要。「3つの」という明確な数を指定しているため、複数の項目が求められている。未来の予測だが、現在のAI開発動向や研究から推測できる範囲であり、確定的な評価が可能。特定の側面について言及しておらず、トレンドの列挙が主な要件なので、複数性のほうが重要。
</think>
<output>
{"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": true,
"needsCompleteness": false}
</output>
</example-6>

<example-7>
Was sind die besten Strategien für nachhaltiges Investieren in der heutigen Wirtschaft?
<think>
Diese Frage bezieht sich auf die "heutige Wirtschaft", daher sind aktuelle Informationen erforderlich. "Strategien" steht im Plural, was auf die Notwendigkeit mehrerer Beispiele hindeutet. Obwohl "beste" subjektiv klingen mag, lässt sich die Frage anhand von Rendite-Daten, Risikobewertungen und Nachhaltigkeitskriterien definitiv evaluieren. Es werden keine spezifischen Aspekte genannt, die alle behandelt werden müssen - der Fokus liegt auf der Vielfalt der Strategien.
</think>
<output>
{"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": true,
"needsCompleteness": false}
</output>
</example-7>

<example-8>
请解释赤壁之战的历史背景、主要参与者以及战略意义，这对中国历史产生了什么影响？
<think>
这是关于历史事件的问题，不需要最新信息。问题明确列出了多个需要评估的方面：历史背景、主要参与者、战略意义和历史影响。这些都是特定的历史课题，需要完整地覆盖。关于历史事件的问题有明确的史料记载和学术研究，可以进行确定性评估。
</think>
<output>
{"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true}
</output>
</example-8>

<example-9>
How might artificial intelligence transform the healthcare industry over the next decade?
<think>
This question asks about future developments that require recent AI and healthcare information to properly evaluate. The query is about transformations (plural) suggesting multiple changes should be identified. While it concerns future trends, the patterns can be evaluated definitively based on current research, development pipelines, and industry analyses. No specific aspects of healthcare are mentioned that must all be addressed, so plurality is more relevant than completeness.
</think>
<output>
{"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": true,
"needsCompleteness": false}
</output>
</example-9>

<example-10>
What do you think is the best approach to learning a new language as an adult?
<think>
This question about language learning does not require time-sensitive information as fundamental language acquisition principles remain relatively stable. The phrasing suggests looking for multiple elements or methods that constitute an effective approach, indicating plurality. Though the question uses subjective language ("what do you think"), it can be evaluated definitively using linguistic research and proven methodologies. No specific language learning aspects are explicitly named that must all be covered.
</think>
<output>
{"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": true,
"needsCompleteness": false}
</output>
</example-10>

<example-11>
If a tree falls in a forest with absolutely no observers, instruments, or any possible way to detect it, does it make a sound?
<think>
This is a classic philosophical paradox that is inherently unanswerable in a definitive way. The question deliberately constructs a scenario that removes all possible means of verification, making it logically impossible to evaluate. This kind of question represents one of the rare cases where a definitive evaluation is truly impossible. The question doesn't involve recent events, doesn't request multiple items, and doesn't specify multiple elements that must be addressed.
</think>
<output>
{"needsDefinitive": false,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": false}
</output>
</example-11>
</examples>

""",
        "user": f"""
{question}
<think>
    """
    }


async def evaluate_question(question, trackers):
    log = get_logger("evaluate_question")
    try:
        generator = ObjectGeneratorSafe(trackers.tokenTracker)
        prompt = get_question_evaluation_prompt(question)
        schemas_gen = None
        if TOOL_NAME == 'evaluator':
            schemas_gen = QuestionEvaluateSchema
        result = await generator.generate_object({
            "model": TOOL_NAME,
            "schema": schemas_gen,
            "system": prompt.get("system"),
            "prompt": prompt.get("user"),
        })
        obj = result["object"]

        types = []
        if obj.get("needsDefinitive"):
            types.append("definitive")
        if obj.get("needsFreshness"):
            types.append("freshness")
        if obj.get("needsPlurality"):
            types.append("plurality")
        if obj.get("needsCompleteness"):
            types.append("completeness")
        log.info(TOOL_NAME + ": {" + question + " " + str(types) + "}")
        trackers.actionTracker.track_think(obj.get("think"))
        return types
    except Exception as e:
        log.error(f"'Error in question evaluation:', {str(e)}")
        return []
