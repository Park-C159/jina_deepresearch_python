# Jina Python

## Introduction

Jina Python is a lightweight, partial Python translation of the open-source Node.js project deep-research. The goal is to make the core research-agent logic accessible to Python users for learning, experimentation, and extension. This is not an official port and it does not include any API endpoints, web servers, or interface handling.

What this project includes

Core research loop: task decomposition, iterative querying, evidence collection, note-taking, and synthesis.
Modular components: replaceable adapters for search, browsing/content retrieval, and storage.
Simple local persistence: optional saving of notes, citations, and intermediate artifacts to the filesystem.
Minimal command-line usage for running the core workflow end-to-end.

What this project does not include (by design)

No HTTP/REST/GraphQL/WebSocket interfaces, and no UI.
No production scaffolding: no auth, sessions, job queues, orchestration, or cloud deployment scripts.
No 1:1 feature parity with the original Node.js codebase; some behaviors are adapted for Python idioms.
Intended audience and use cases

Python developers who want to study or extend the deep-research agentic workflow.
Researchers and tinkerers who prefer Python tooling for quick iteration and offline experiments.
Educational settings where a concise, readable Python codebase is helpful.
Status and scope

Work-in-progress; functionality is translated to a practical extent but not exhaustively.
Differences from the Node.js version may exist due to language semantics and library choices.
Contributions, issue reports, and suggestions are welcome.