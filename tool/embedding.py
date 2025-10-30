import os

import requests
import logging
import time

from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 32
API_URL = "https://api.jina.ai/v1/embeddings"
MAX_RETRIES = 3
JINA_API_KEY = os.getenv("JINA_API_KEY")


def trim_symbols(s: str) -> str:
    import regex  # ⚠️ 注意这里是 regex，不是 re
    # 去除符号 (S)、标点 (P)、空白 (Z)、控制字符 (C)
    regex_pattern = r'[\p{S}\p{P}\p{Z}\p{C}]+'
    return regex.sub(regex_pattern, ' ', s)



def truncate_input_string(input_):
    if isinstance(input_, str):
        return input_[:50]
    elif isinstance(input_, dict):
        return list(input_.values())[0][:50]
    return str(input_)[:50]


def get_embeddings(
        texts,
        token_tracker=None,
        options=None
):
    logging.debug(f"[embeddings] Getting embeddings for {len(texts)} texts")
    if not JINA_API_KEY:
        raise Exception("JINA_API_KEY is not set")

    # 预处理空输入
    if len(texts) == 0 or all(
            (isinstance(text, dict) and not list(text.values())[0].strip())
            or (isinstance(text, str) and not text.strip())
            for text in texts):
        return {"embeddings": [], "tokens": 0}

    options = options or {}
    all_embeddings = []
    total_tokens = 0
    batch_count = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        current_batch = i // BATCH_SIZE + 1
        logging.debug(f"Embedding batch {current_batch}/{batch_count} ({len(batch_texts)} texts)")
        batch_embeddings, batch_tokens = get_batch_embeddings_with_retry(
            batch_texts, options, current_batch, batch_count
        )
        all_embeddings.extend(batch_embeddings)
        total_tokens += batch_tokens
        logging.debug(
            f"[embeddings] Batch {current_batch} complete. Tokens used: {batch_tokens}, total so far: {total_tokens}")

    if token_tracker:
        token_tracker.track_usage('embeddings', {
            'promptTokens': total_tokens,
            'completionTokens': 0,
            'totalTokens': total_tokens,
        })
    logging.debug(f"[embeddings] Complete. Generated {len(all_embeddings)} embeddings using {total_tokens} tokens")
    return {"embeddings": all_embeddings, "tokens": total_tokens}


def get_batch_embeddings_with_retry(batch_texts, options, current_batch, batch_count):
    batch_embeddings = []
    batch_tokens = 0
    retry_count = 0

    texts_to_process = []
    index_map = {}
    for idx, item in enumerate(batch_texts):
        if isinstance(item, str):
            texts_to_process.append(trim_symbols(item))
        else:
            key = list(item.keys())[0]
            text = item[key]
            if key == 'text':
                texts_to_process.append({'text': trim_symbols(text)})
            else:
                texts_to_process.append(item)
        index_map[idx] = idx

    while len(texts_to_process) > 0 and retry_count < MAX_RETRIES:
        request_json = {
            "model": options.get("model", "jina-embeddings-v3"),
            "input": texts_to_process,
        }
        if request_json["model"] == "jina-embeddings-v3":
            request_json["task"] = options.get("task", "text-matching")
            request_json["truncate"] = True
        if options.get("dimensions"):
            request_json["dimensions"] = options["dimensions"]
        if options.get("late_chunking"):
            request_json["late_chunking"] = options["late_chunking"]
        if options.get("embedding_type"):
            request_json["embedding_type"] = options["embedding_type"]

        try:
            response = requests.post(
                API_URL,
                json=request_json,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {JINA_API_KEY}",
                },
                timeout=60,
            )
            response.raise_for_status()  # 抛异常给 except
            resp_data = response.json()
            if "data" not in resp_data or not resp_data["data"]:
                logging.error("No data returned from Jina API")
                if retry_count == MAX_RETRIES - 1:
                    dimension_size = options.get("dimensions", 1024)
                    placeholder_embeddings = [
                        [0] * dimension_size for _ in texts_to_process
                    ]
                    for i in range(len(texts_to_process)):
                        original_idx = index_map[i]
                        while len(batch_embeddings) <= original_idx:
                            batch_embeddings.append([])
                        batch_embeddings[original_idx] = placeholder_embeddings[i]
                retry_count += 1
                continue

            received_indices = set(item["index"] for item in resp_data["data"])
            dimension_size = resp_data["data"][0]["embedding"] and len(
                resp_data["data"][0]["embedding"]) or options.get("dimensions", 1024)

            successful_embeddings = []
            remaining_texts = []
            new_index_map = {}

            for idx in range(len(texts_to_process)):
                if idx in received_indices:
                    item = next(d for d in resp_data["data"] if d["index"] == idx)
                    original_idx = index_map[idx]
                    while len(batch_embeddings) <= original_idx:
                        batch_embeddings.append([])
                    batch_embeddings[original_idx] = item["embedding"]
                    successful_embeddings.append(item["embedding"])
                else:
                    # retry
                    new_index = len(remaining_texts)
                    new_index_map[new_index] = index_map[idx]
                    remaining_texts.append(texts_to_process[idx])
                    logging.warning(
                        f"Missing embedding for index {idx}, will retry: [{truncate_input_string(texts_to_process[idx])}...]")

            batch_tokens += resp_data.get("usage", {}).get("total_tokens", 0)
            texts_to_process = remaining_texts
            index_map = new_index_map

            if len(texts_to_process) == 0:
                break
            retry_count += 1
            logging.debug(
                f"[embeddings] Batch {current_batch}/{batch_count} - Retrying {len(texts_to_process)} texts (attempt {retry_count}/{MAX_RETRIES})")
        except Exception as error:
            logging.error(f"Error calling Jina Embeddings API: {error}")
            status = getattr(error.response, 'status_code', None)
            if status == 402 or "InsufficientBalanceError" in str(error) or "insufficient balance" in str(error):
                return [], 0
            if retry_count == MAX_RETRIES - 1:
                dimension_size = options.get("dimensions", 1024)
                for idx in range(len(texts_to_process)):
                    original_idx = index_map[idx]
                    logging.error(
                        f"Failed to get embedding after all retries for index {original_idx}: [{truncate_input_string(texts_to_process[idx])}...]")
                    while len(batch_embeddings) <= original_idx:
                        batch_embeddings.append([])
                    batch_embeddings[original_idx] = [0] * dimension_size
            retry_count += 1
            if retry_count < MAX_RETRIES:
                logging.debug(
                    f"[embeddings] Batch {current_batch}/{batch_count} - Retry attempt {retry_count}/{MAX_RETRIES} after error")
                time.sleep(1)  # 避免打爆API
            else:
                raise error

    # 最后处理所有失败的 embedding
    if len(texts_to_process) > 0:
        logging.error(
            f"[embeddings] Failed to get embeddings for {len(texts_to_process)} texts after {MAX_RETRIES} retries")
        dimension_size = options.get("dimensions", 1024)
        for idx in range(len(texts_to_process)):
            original_idx = index_map[idx]
            logging.error(f"Creating zero embedding for index {original_idx} after all retries failed")
            while len(batch_embeddings) <= original_idx:
                batch_embeddings.append([])
            batch_embeddings[original_idx] = [0] * dimension_size

    return batch_embeddings, batch_tokens

# 使用方法示例
# result = get_embeddings(['hello world', '你好，世界'], options={"dimensions": 1024})
# print(result)
