# Jina DeepResearch Agent (Python Version)

## Introduction

This project is a Python async implementation of the Jina AI DeepResearch model, converted from its original TypeScript version.
It serves as an intelligent research agent that retrieves, filters, and analyzes information across multiple sources to produce high-quality research summaries or answers.


## Features

- Multi-source information retrieval using Jina AI or other custom search providers

- Multi-language search and response support

- Generates structured research outputs saved to result_output.txt

- Optional image understanding capabilities

- Hostname boosting and penalization for search control

- Fully configurable parameters and parallel processing (team mode)

## Requirements

- Python ≥ 3.9

- Async support (asyncio)

- Recommended to use a virtual environment

Install dependencies:

``` shell
pip install -r requirements.txt
```

## Usage
Main entry point: agent.py

### Example command

```shell
python agent.py \
  --question "Explain how diffusion models work" \
  --search_provider jina \
  --language_code en \
  --with_images \
  --token_budget 200000 \
  --num_returned_urls 5 \
  --boost_hostnames "huggingface.co" "arxiv.org"
```

### Parameter Description

| Parameter                | Type        | Default      | Description                                                  |
| ------------------------ | ----------- | ------------ | ------------------------------------------------------------ |
| `--question`             | `str`       | **Required** | The user’s input question                                    |
| `--search_language_code` | `str`       | `"en"`       | Language code used for search (e.g., `"zh"`, `"en"`, `"fr"`) |
| `--search_provider`      | `str`       | `"jina"`     | Search provider, e.g. `"jina"` or `"none"` to disable search |
| `--language_code`        | `str`       | `"en"`       | Response language code                                       |
| `--with_images`          | `flag`      | `False`      | Enable image content analysis                                |
| `--token_budget`         | `int`       | `100000000`  | Maximum token usage limit                                    |
| `--max_bad_attempts`     | `int`       | `2`          | Maximum number of failed attempts before stopping            |
| `--existing_context`     | `str`       | `None`       | Existing context for multi-turn research tracking            |
| `--num_returned_urls`    | `int`       | `5`          | Number of URLs returned from search                          |
| `--no_direct_answer`     | `flag`      | `False`      | Disable direct answering mode (report generation only)       |
| `--boost_hostnames`      | `list[str]` | `[]`         | Domains to prioritize (e.g., trusted sources)                |
| `--bad_hostnames`        | `list[str]` | `[]`         | Domains to penalize                                          |
| `--only_hostnames`       | `list[str]` | `None`       | Restrict results to specific domains only                    |
| `--max_ref`              | `int`       | `50`         | Maximum number of references                                 |
| `--min_rel_score`        | `float`     | `0.7`        | Minimum relevance score threshold                            |
| `--team_size`            | `int`       | `1`          | Number of agents for parallel (team) processing              |

## Output
All results are appended to a local file:

result_output.txt


Each call to get_response() produces a structured JSON output, for example:

```json
{
  "result": "Deep explanation about diffusion models...",
  "references": ["https://arxiv.org/abs/2102.09672", "..."],
  "metadata": {
    "query": "Explain how diffusion models work",
    "language": "en",
    "token_used": 3274
  }
}
```

The main summary will also be printed to the console:

``` shell
"result": "Diffusion models generate data by iteratively denoising random noise..."
```