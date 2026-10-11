# Model and release governance

Owner: local operator; assign a named accountable owner before deployment.

Provider: Groq, configurable via GROQ_MODEL. The LLM selects facts, never financial formulas. Review provider retention and no-training terms before sending sensitive holdings.

Validation evidence: automated accounting, ingestion, split, XIRR, benchmark, AI guardrail and Streamlit tests in tests/. Fixtures are synthetic, independently specified expectations, not Excel-certified or broker-reconciled.

Monitoring: session request/token budgets, tool limits, local execution metadata, token and latency counts, session thumbs feedback. No persistent telemetry or transaction traces. Dollar cost is not claimed because model prices change; review provider usage separately.

Open launch gates:
- Human review of formulas, prompt and verifier.
- Ten Excel-certified XIRR portfolios and three real broker reconciliations.
- Full PRD 500-question AI evaluation, 50-session trace review and measured latency/cost budgets.
- Privacy/security/legal/model-risk approvals and provider contract review.
- No open high dependency vulnerabilities; licensed production market feed.
- Accessibility review and complete corporate-action reconciliation.

Do not claim these reviews have passed or that this build is production-approved.
