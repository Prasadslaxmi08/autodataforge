# Security Policy

## Supported Versions

AutoDataForge is pre-1.0. Security fixes are applied to the latest `main` and the
most recent tagged release.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a Vulnerability

**Please do not open a public issue for security vulnerabilities.**

Instead, report privately via one of:

- GitHub's [private vulnerability reporting](https://github.com/Prasadslaxmi08/autodataforge/security/advisories/new)
- Email: **eonspacelabspvtltd@gmail.com** with subject `SECURITY: AutoDataForge`

Include:

- A description of the vulnerability and its impact
- Steps to reproduce (proof-of-concept if possible)
- Affected version / commit

We aim to acknowledge reports within **72 hours** and to provide a remediation
timeline after triage. We will credit reporters in the release notes unless you
prefer to remain anonymous.

## Scope notes

AutoDataForge runs models and processes user-supplied media and video files. When
deploying:

- Treat imported files as untrusted input.
- The MCP server confines file access to configured paths — keep that boundary.
- Model weights are downloaded from third-party sources; pin and verify them in
  production environments.
