import math
import re
from typing import Any, Dict, List, Tuple

from tool.cosin import cosine_similarity, jaccard_rank
from tool.embedding import get_embeddings
from tool.image_tools import dedup_images_with_embeddings
from tool.segments import chunk_text
from utils.schemas import LANGUAGE_CODE
from utils.url_tool import normalize_host_name


# === 日志辅助 ===
def log_debug(*args, **kwargs): print("[DEBUG]", *args, kwargs)


def log_error(*args, **kwargs): print("[ERROR]", *args, kwargs)


# === buildReferences 主函数 ===
async def build_references(
        answer,
        web_contents,
        context,
        min_chunk_length=80,
        max_ref=10,
        min_rel_score=0.7,
        only_hostnames=[]
):
    log_debug(
        f"[buildReferences] Starting with maxRef={max_ref}, minChunkLength={min_chunk_length}, minRelScore={min_rel_score}")
    log_debug(f"[buildReferences] Answer length: {len(answer)} chars, Web content sources: {len(web_contents)}")

    # Step 1
    log_debug("[buildReferences] Step 1: Chunking answer text")
    result = chunk_text(answer)
    answer_chunks = result["chunks"]
    answer_positions = result["chunk_positions"]
    log_debug(f"[buildReferences] Answer segmented into {len(answer_chunks)} chunks")

    # Step 2
    log_debug(
        f"[buildReferences] Step 2: Preparing web content chunks and filtering by minimum length ({min_chunk_length})")
    all_web_chunks = []
    chunk_to_source = {}
    valid_web_indices = set()
    idx = 0
    for url, content in web_contents.items():
        if not content.get("chunks"): continue
        if only_hostnames and normalize_host_name(url) not in only_hostnames:
            continue
        for chunk in content["chunks"]:
            all_web_chunks.append(chunk)
            chunk_to_source[idx] = {"url": url, "title": content.get("title", url), "text": chunk}
            if len(chunk) >= min_chunk_length:
                valid_web_indices.add(idx)
            idx += 1
    log_debug(
        f"[buildReferences] Collected {len(all_web_chunks)} web chunks, {len(valid_web_indices)} above minimum length")

    if not all_web_chunks:
        return {"answer": answer, "references": []}

    # Step 3
    log_debug("[buildReferences] Step 3: Filtering answer chunks by minimum length")

    context.actionTracker.track_think('cross_reference', LANGUAGE_CODE)

    valid_answer_chunks = []
    valid_answer_indices = []
    valid_positions = []

    for i, chunk in enumerate(answer_chunks):
        pos = answer_positions[i]
        if not chunk.strip() or len(chunk) < min_chunk_length:
            continue
        valid_answer_chunks.append(chunk)
        valid_answer_indices.append(i)
        valid_positions.append(pos)

    log_debug(
        f"[buildReferences] Found {len(valid_answer_chunks)}/{len(answer_chunks)} valid answer chunks above minimum length")

    if not valid_answer_chunks:
        return {"answer": answer, "references": []}

    # Step 4
    log_debug("[buildReferences] Step 4: Getting embeddings for all chunks in a single request")
    all_chunks = []
    index_map = {}
    for i, c in enumerate(valid_answer_chunks):
        all_chunks.append(c)
        index_map[len(all_chunks) - 1] = {"type": "answer", "idx": i}
    for i, c in enumerate(all_web_chunks):
        if i in valid_web_indices:
            all_chunks.append(c)
            index_map[len(all_chunks) - 1] = {"type": "web", "idx": i}

    log_debug(f"[buildReferences] Requesting embeddings for {len(all_chunks)} total chunks")
    try:
        res = get_embeddings(all_chunks)
        all_emb = res["embeddings"]
        answer_emb = []
        web_emb_map = {}

        for i, emb in enumerate(all_emb):
            mapping = index_map[i]
            if mapping["type"] == "answer":
                answer_emb.append(emb)
            else:
                web_emb_map[mapping["idx"]] = emb

        log_debug(f"[buildReferences] Got embeddings: {len(answer_emb)} answer, {len(web_emb_map)} web")

        # Step 5
        log_debug("[buildReferences] Step 5: Computing cosine similarity")
        all_matches = []
        for i, chunk in enumerate(valid_answer_chunks):
            a_idx = valid_answer_indices[i]
            a_pos = valid_positions[i]
            a_emb = answer_emb[i]
            for w_idx in valid_web_indices:
                w_emb = web_emb_map.get(w_idx)
                if not w_emb: continue
                score = cosine_similarity(a_emb, w_emb)
                all_matches.append({
                    "webChunkIndex": w_idx,
                    "answerChunkIndex": a_idx,
                    "relevanceScore": score,
                    "answerChunk": chunk,
                    "answerChunkPosition": a_pos
                })
        all_matches.sort(key=lambda x: x["relevanceScore"], reverse=True)
        log_debug(f"[buildReferences] Step 6: Sorted {len(all_matches)} matches")

        used_web = set()
        used_ans = set()
        filtered = []
        for m in all_matches:
            if m["relevanceScore"] < min_rel_score:
                continue
            if m["webChunkIndex"] in used_web or m["answerChunkIndex"] in used_ans:
                continue
            filtered.append(m)
            used_web.add(m["webChunkIndex"])
            used_ans.add(m["answerChunkIndex"])
            if len(filtered) >= max_ref:
                break
        log_debug(f"[buildReferences] Selected {len(filtered)} matches after filtering")

        return build_final_result(answer, filtered, chunk_to_source)

    except Exception as e:
        log_error("Embedding failed, falling back to Jaccard similarity", e)
        all_matches = []
        for i, chunk in enumerate(valid_answer_chunks):
            a_idx = valid_answer_indices[i]
            a_pos = valid_positions[i]
            res = jaccard_rank(chunk, all_web_chunks)
            for m in res["results"]:
                if m["index"] in valid_web_indices:
                    all_matches.append({
                        "webChunkIndex": m["index"],
                        "answerChunkIndex": a_idx,
                        "relevanceScore": m["relevance_score"],
                        "answerChunk": chunk,
                        "answerChunkPosition": a_pos
                    })
        all_matches.sort(key=lambda x: x["relevanceScore"], reverse=True)
        used_web = set()
        used_ans = set()
        filtered = []
        for m in all_matches:
            if m["relevanceScore"] < min_rel_score:
                continue
            if m["webChunkIndex"] in used_web or m["answerChunkIndex"] in used_ans:
                continue
            filtered.append(m)
            used_web.add(m["webChunkIndex"])
            used_ans.add(m["answerChunkIndex"])
            if len(filtered) >= max_ref:
                break
        log_debug(f"[buildReferences] Selected {len(filtered)} fallback references")
        return build_final_result(answer, filtered, chunk_to_source)


def build_final_result(answer, filtered_matches, chunk_to_source):
    log_debug(f"[buildFinalResult] Building final result with {len(filtered_matches)} references")
    references = []
    for m in filtered_matches:
        s = chunk_to_source[m["webChunkIndex"]]
        if not s["text"] or not s["url"] or not s["title"]:
            continue
        references.append({
            "exactQuote": s["text"],
            "url": s["url"],
            "title": s["title"],
            "relevanceScore": m["relevanceScore"],
            "answerChunk": m["answerChunk"],
            "answerChunkPosition": m["answerChunkPosition"]
        })

    modified = answer
    refs_by_pos = sorted(references, key=lambda r: r["answerChunkPosition"][0])
    offset = 0
    for i, ref in enumerate(refs_by_pos):
        marker = f"[^{i + 1}]"
        pos = ref["answerChunkPosition"][1] + offset
        modified = modified[:pos] + marker + modified[pos:]
        offset += len(marker)

    log_debug(f"[buildFinalResult] Complete. Generated {len(references)} references")
    return {"answer": modified, "references": references}


# === buildImageReferences ===
async def build_image_references(
        answer,
        image_objects,
        context,
        min_chunk_length=80,
        max_ref=10,
        min_rel_score=0.35
):
    log_debug(f"[buildImageReferences] Starting with maxRef={max_ref}, "
              f"minChunkLength={min_chunk_length}, minRelScore={min_rel_score}")
    log_debug(f"[buildImageReferences] Answer length: {len(answer)} chars, "
              f"Image sources: {len(image_objects)}")

    # Step 1: Chunk answer
    log_debug("[buildImageReferences] Step 1: Chunking answer text")
    chunk_result = chunk_text(answer)
    answer_chunks = chunk_result["chunks"]
    answer_chunk_positions = chunk_result["chunk_positions"]
    log_debug(f"[buildImageReferences] Answer segmented into {len(answer_chunks)} chunks")

    # Step 2: Prepare image content
    log_debug("[buildImageReferences] Step 2: Preparing image content")
    dedup_images = dedup_images_with_embeddings(image_objects, [])
    all_image_embeddings = [img["embedding"][0] for img in dedup_images]
    image_to_source_map = {}
    valid_image_indices = set()

    for idx, img in enumerate(dedup_images):
        image_to_source_map[idx] = {
            "url": img.get("url"),
            "altText": img.get("alt"),
            "embedding": img["embedding"][0],
        }
        valid_image_indices.add(idx)

    log_debug(f"[buildImageReferences] Collected {len(all_image_embeddings)} image embeddings")

    if not all_image_embeddings:
        log_debug("[buildImageReferences] No image data available, returning empty array")
        return []

    # Step 3: Filter answer chunks
    log_debug("[buildImageReferences] Step 3: Filtering answer chunks by minimum length")
    valid_answer_chunks = []
    valid_answer_chunk_indices = []
    valid_answer_chunk_positions = []

    context.action_tracker.track_think("cross_reference", LANGUAGE_CODE)

    for i, chunk in enumerate(answer_chunks):
        position = answer_chunk_positions[i]
        if not chunk.strip() or len(chunk) < min_chunk_length:
            continue
        valid_answer_chunks.append(chunk)
        valid_answer_chunk_indices.append(i)
        valid_answer_chunk_positions.append(position)

    log_debug(f"[buildImageReferences] Found {len(valid_answer_chunks)}/{len(answer_chunks)} "
              f"valid answer chunks above minimum length")

    if not valid_answer_chunks:
        log_debug("[buildImageReferences] No valid answer chunks, returning empty array")
        return []

    # Step 4: Get embeddings
    log_debug("[buildImageReferences] Step 4: Getting embeddings for answer chunks")
    answer_embeddings = []

    try:
        embeddings_result = await get_embeddings(
            valid_answer_chunks,
            context.token_tracker,
            {"dimensions": 512, "model": "jina-clip-v2"}
        )
        answer_embeddings.extend(embeddings_result["embeddings"])
        log_debug(f"[buildImageReferences] Got embeddings for {len(answer_embeddings)} answer chunks")

        # Step 5: Compute cosine similarity
        log_debug(
            "[buildImageReferences] Step 5: Computing pairwise cosine similarity between answer and image embeddings")
        all_matches = []

        for i, answer_chunk in enumerate(valid_answer_chunks):
            answer_chunk_index = valid_answer_chunk_indices[i]
            answer_chunk_position = valid_answer_chunk_positions[i]
            answer_embedding = answer_embeddings[i]
            matches_for_chunk = []

            for image_index in valid_image_indices:
                image_embedding = all_image_embeddings[image_index]
                if image_embedding:
                    score = cosine_similarity(answer_embedding, image_embedding)
                    matches_for_chunk.append({
                        "imageIndex": image_index,
                        "relevanceScore": score
                    })

            matches_for_chunk.sort(key=lambda x: x["relevanceScore"], reverse=True)
            for match in matches_for_chunk:
                all_matches.append({
                    "imageIndex": match["imageIndex"],
                    "answerChunkIndex": answer_chunk_index,
                    "relevanceScore": match["relevanceScore"],
                    "answerChunk": answer_chunk,
                    "answerChunkPosition": answer_chunk_position
                })

            top_score = matches_for_chunk[0]["relevanceScore"] if matches_for_chunk else 0
            log_debug(f"[buildImageReferences] Processed answer chunk {i + 1}/{len(valid_answer_chunks)}, "
                      f"top score: {top_score:.4f}")

        # 统计分析
        if all_matches:
            scores = [m["relevanceScore"] for m in all_matches]
            stats = {
                "min": f"{min(scores):.4f}",
                "max": f"{max(scores):.4f}",
                "mean": f"{sum(scores) / len(scores):.4f}",
                "count": len(scores)
            }
            log_debug("Reference relevance statistics:", stats)

        # Step 6: Sort by relevance
        all_matches.sort(key=lambda x: x["relevanceScore"], reverse=True)
        log_debug(f"[buildImageReferences] Step 6: Sorted {len(all_matches)} potential matches")

        # Step 7: Filter matches
        log_debug(f"[buildImageReferences] Step 7: Filtering matches (min: {min_rel_score})")
        used_images = set()
        used_answer_chunks = set()
        filtered_matches = []

        for match in all_matches:
            # if match["relevanceScore"] < min_rel_score:
            #     continue
            if match["imageIndex"] not in used_images and match["answerChunkIndex"] not in used_answer_chunks:
                filtered_matches.append(match)
                used_images.add(match["imageIndex"])
                used_answer_chunks.add(match["answerChunkIndex"])
                if len(filtered_matches) >= max_ref:
                    break

        log_debug(
            f"[buildImageReferences] Selected {len(filtered_matches)}/{len(all_matches)} references after filtering")

        references = []
        for match in filtered_matches:
            source = image_to_source_map[match["imageIndex"]]
            references.append({
                "url": source["url"],
                "relevanceScore": match["relevanceScore"],
                "embedding": [all_image_embeddings[match["imageIndex"]]],
                "answerChunk": match["answerChunk"],
                "answerChunkPosition": match["answerChunkPosition"]
            })

        return references

    except Exception as error:
        log_error("Embedding failed", {"error": error})
        return []
