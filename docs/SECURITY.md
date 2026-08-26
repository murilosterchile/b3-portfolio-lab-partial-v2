# Security baseline

This repository is safe for local research by default, not a claim of production certification.

## Implemented

- Argon2id password hashing.
- Server-side session records; plaintext session tokens are never stored in PostgreSQL.
- HttpOnly session cookie and separate CSRF cookie.
- Double-submit/session-bound CSRF validation on protected writes.
- SameSite cookies, Secure cookies/HSTS outside development.
- CORS allowlist.
- Security headers on both API and Next.js UI, including a restrictive baseline CSP.
- Distributed fixed-window rate limiting through Valkey for authentication and portfolio optimization.
- The optimizer requires an authenticated session + CSRF whenever `DEMO_MODE=false`.
- Non-root long-running API and web processes.
- Secrets configured through environment variables; `.env` is ignored.
- CI hooks for lint/tests plus dependency/secret/filesystem security scanning.

## Before exposing to the Internet

- Generate a high-entropy `SECRET_KEY`; never use the example value.
- Terminate TLS with a maintained reverse proxy/load balancer and tighten CSP to the deployed origins.
- Persist immutable audit events for authentication and every generated recommendation/portfolio.
- Back up PostgreSQL and regularly test restores.
- Move secrets to a managed secret store and define rotation procedures.
- Pin and attest release dependencies/container images; generate SBOMs for releases.
- Review LGPD data minimization/retention and create deletion/export flows.
- Add account lockout/risk-based controls if public authentication is enabled.
- Commission an external security review/penetration test.
