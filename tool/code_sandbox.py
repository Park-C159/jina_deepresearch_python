import datetime
import logging
import textwrap

from utils.safe_generator import ObjectGeneratorSafe
from utils.schemas import CodeGeneratorSchema


def get_prompt(problem, available_vars, previous_attempts):
    previous_attempts_arr = []
    for attempt, index in previous_attempts:
        previous_attempt_str = f"""
<bad-attempt-${index + 1}>
{attempt.get('code')}
{'Error: ' + str(attempt.get('error')) if attempt.get('error') else attempt.get('error')}
</bad-attempt-${index + 1}>
"""
        previous_attempts_arr.append(previous_attempt_str)
    previous_attempts_context = '\n'.join(previous_attempts_arr)
    prompt = f"""You are an expert JavaScript programmer. Your task is to generate JavaScript code to solve the given problem.

<rules>
1. Generate plain JavaScript code that returns the result directly
2. You can access any of these available variables directly:
{available_vars}
3. You don't have access to any third party libraries that need to be installed, so you must write complete, self-contained code.
4. Must have a return statement.
</rules>

{'Previous attempts and their errors:' + str(previous_attempts_context) if len(previous_attempts) > 0 else ''}

<example>
Available variables:
numbers (Array<number>) e.g. [1, 2, 3, 4, 5, 6]
threshold (number) e.g. 4

Problem: Sum all numbers above threshold

Response:
{{
    "code": "return numbers.filter(n => n > threshold).reduce((a, b) => a + b, 0);"
}}
</example>"""
    logging.debug('Coding prompt', {prompt})

    return {
        'system': prompt,
        'user': problem
    }


class CodeSandbox:
    def __init__(self, context=None, trackers=None,  maxAttempts=3):
        self.trackers = trackers
        token_tracker = getattr(trackers, 'tokenTracker', None) if trackers else None

        # 兼容外部依赖命名：ObjectGeneratorSafe 必须由外部提供
        # 若不存在，会在实际调用时抛错（保持与原始代码一致的依赖要求）
        self.generator = ObjectGeneratorSafe(token_tracker)

        self.maxAttempts = maxAttempts
        self.context = context if isinstance(context, dict) else (context or {})
        self.schemaGen = CodeGeneratorSchema

    async def generateCode(self, problem, previousAttempts=None):
        if previousAttempts is None:
            previousAttempts = []

        # 兼容外部函数命名：getPrompt 必须由外部提供
        prompt = get_prompt(problem, analyzeStructure(self.context), previousAttempts)

        schema = self.schemaGen if self.schemaGen else None

        # 兼容方法命名：generate_object / generateObject
        gen_method = getattr(self.generator, 'generate_object', None) or getattr(self.generator, 'generateObject', None)
        if not gen_method:
            raise RuntimeError('Code generator method not found (generate_object / generateObject).')

        result = await gen_method({
            'model': 'coder',
            'schema': schema,
            'system': prompt.get('system'),
            'prompt': prompt.get('user'),
        })

        # 记录思考轨迹（若可用）
        action_tracker = getattr(self.trackers, 'actionTracker', None) if self.trackers else None
        track_think = getattr(action_tracker, 'trackThink', None) or getattr(action_tracker, 'track_think', None)
        if track_think:
            obj = result.get('object') if isinstance(result, dict) else None
            think = (obj or {}).get('think') if isinstance(obj, dict) else None
            track_think(think)

        # 返回对象部分
        obj = result.get('object') if isinstance(result, dict) else None
        return obj

    def evaluateCode(self, code):
        try:
            # 将用户代码包裹进函数体，要求显式 return
            func_code = "def __eval_in_context():\n" + textwrap.indent(code, "    ") + "\n"

            # 在 context 中执行，模拟 JS 的 with(context) {...}
            # 函数调用时会以 env 作为全局命名空间，代码中可直接使用 context 内的键作为变量
            env = dict(self.context) if isinstance(self.context, dict) else {'context': self.context}

            logging.debug('Context:', {'context': self.context})

            exec(func_code, env, env)
            output = env['__eval_in_context']()

            if output is None:
                return {
                    'success': False,
                    'error': 'No value was returned, make sure to use "return" statement to return the result'
                }

            return {
                'success': True,
                'output': output
            }
        except Exception as error:
            return {
                'success': False,
                'error': str(error) if str(error) else 'Unknown error occurred'
            }

    async def solve(self, problem):
        attempts = []

        for i in range(self.maxAttempts):
            # 生成代码
            generation = await self.generateCode(problem, attempts)
            code = (generation or {}).get('code') if isinstance(generation, dict) else None

            # 防御：若生成阶段没有返回 code
            if not code:
                error_msg = 'No code was generated'
                logging.warning('Coding error:', {'error': error_msg})
                attempts.append({'code': '', 'error': error_msg})
                if i == self.maxAttempts - 1:
                    raise RuntimeError(f'Failed to generate working code after {self.maxAttempts} attempts')
                continue

            # 执行并评估
            result = self.evaluateCode(code)

            if result.get('success'):
                logging.info('Coding success:', {'problem': problem, 'result': result})
                return {
                    'solution': {
                        'code': code,
                        'output': result.get('output')
                    },
                    'attempts': attempts
                }

            # 记录失败并继续迭代
            logging.warning('Coding error:', {'error': result.get('error')})
            attempts.append({
                'code': code,
                'error': result.get('error')
            })

            if i == self.maxAttempts - 1:
                raise RuntimeError(f'Failed to generate working code after {self.maxAttempts} attempts')

        # 理论上不会到这里
        raise RuntimeError('Unexpected end of execution')


def formatValue(value):
    if value is None:
        return 'null'

    if isinstance(value, str):
        cleaned = value.replace('\n', ' ').strip()
        # 折叠多空白
        cleaned = ' '.join(cleaned.split())
        return f"\"{cleaned[:47]}...\"" if len(cleaned) > 50 else f"\"{cleaned}\""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, datetime.datetime):
        return f"\"{value.isoformat()}\""

    return ''


def analyzeStructure(value, indent=''):
    if value is None:
        return 'null'

    # 函数
    if callable(value):
        return 'Function'

    # 原子类型与日期（视为原子）
    is_atomic = (
            isinstance(value, (str, bool, int, float)) or
            isinstance(value, datetime.datetime)
    )

    if is_atomic:
        # 模拟 JS typeof 的名称
        if isinstance(value, str):
            t = 'string'
        elif isinstance(value, bool):
            t = 'boolean'
        elif isinstance(value, (int, float)):
            t = 'number'
        elif isinstance(value, datetime.datetime):
            # 与原实现一致：Date 在分支里也归为 atomic，但类型名仍沿用通用 ‘object’ 逻辑
            t = 'object'
        else:
            t = 'object'

        formatted = formatValue(value)
        return f"{t}{f' (example: {formatted})' if formatted else ''}"

    # 数组（列表/元组）
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 'Array<unknown>'
        sample_item = value[0]
        return f"Array<{analyzeStructure(sample_item, indent + '  ')}>"

    # 对象（字典）
    if isinstance(value, dict):
        if len(value) == 0:
            return '{}'
        lines = []
        for k, v in value.items():
            analyzed = analyzeStructure(v, indent + '  ')
            lines.append(f'{indent}  "{k}": {analyzed}')
        return "{\n" + ",\n".join(lines) + f"\n{indent}}}"

    # 其他对象：给出通用 object
    return 'object'
