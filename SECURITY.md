# PV CrackVision — Security posture and deployment notes

## Scope

This application hardens the local desktop application and its local HTTP API. It is not a security certification, independent penetration-test report, or guarantee that the application is free of vulnerabilities. The source archive excludes model binaries and is not a signed Windows installer.

## Implemented controls

- **Local-first binding:** the default server bind is `127.0.0.1`. A non-loopback bind is ignored unless `PV_ALLOW_REMOTE_BIND=1` and `PV_REMOTE_TOKEN` contains at least 32 characters.
- **Browser request validation:** every request is checked against the expected Host port. The API rejects cross-site Fetch Metadata, mismatched/insecure Origins, and non-loopback hosts that do not meet the remote-token policy. DNS-rebinding Host names do not receive the loopback exemption.
- **API abuse controls:** a global per-client fixed-window request limiter is applied to `/api`; request bodies have explicit size limits on relevant routes.
- **Response headers:** CSP, `nosniff`, frame denial, no-referrer, a restrictive Permissions Policy, same-origin opener/resource policies, no-store API responses, and disabled framework fingerprinting.
- **Input validation:** feedback uploads are decoded as supported raster images; annotation counts, coordinate dimensions, rectangles, module outlines, categories and model scores are validated against decoded source-image bounds.
- **Electron isolation:** `contextIsolation`, sandboxing and disabled Node integration; cross-origin top-level navigation and webviews blocked; renderer permissions denied; external links are limited to HTTPS.
- **Persistent-data integrity:** feedback indexes and undo journals use atomic replacement and restrictive POSIX modes where available. Invalid/corrupt indexes are not treated as empty datasets. Image identifiers are constrained before filesystem access. Undo uses durable before/after records and refuses stale overwrites; transactions are bounded to 10,000 changed rows and a 64 MiB journal to avoid unbounded recovery files.

## Required deployment practices

1. Run the application under a dedicated, non-administrator Windows user where practical.
2. Keep the data folder on a locally controlled NTFS volume; restrict its ACL to the intended Windows user and administrators. Do not place it in a broadly shared folder.
3. Back up the complete data folder before release upgrades, training resets, or bulk annotation operations. The undo journal is time- and count-limited; it is not a backup system.
4. Keep Windows, graphics drivers, Node/Electron build dependencies, Python packages and model files updated through a controlled release process. Generate remote tokens from a cryptographically secure random source; do not use a memorable or repeated phrase.
5. Do not expose the local HTTP port to a LAN or the public internet. The optional remote token is not a substitute for TLS, user identity, authorization roles, or a hardened reverse proxy. Remote browser UI operation has not been validated as a supported deployment mode; use the local desktop window.
6. Build and sign the Windows installer in a trusted CI environment. Verify the release artifact hash and signature before distribution.

## Commercial-release gates

Automated in CI (`.github/workflows/windows-desktop.yml`): production dependency audit (`npm audit`, report-only), CycloneDX SBOM, third-party notices drift check, SHA-256 checksums for every deliverable, packaged-app smoke test, and code signing that activates when the protected secrets `CSC_LINK` and `CSC_KEY_PASSWORD` are set.

Still required before commercial distribution (these need the vendor, a Windows runner or a third party):

1. **Legal review of licensing.** Choose the product license/EULA (`package.json` is marked `UNLICENSED`, i.e. proprietary, until you decide). The optional Python training feature imports Ultralytics YOLO (AGPL-3.0; commercial Enterprise license available) — confirm obligations with counsel. See `THIRD_PARTY_NOTICES.md`.
2. Triage the `npm audit` findings and the SBOM produced by CI for each release.
3. Obtain a code-signing certificate and set the CI secrets; verify the signature on the produced EXE files.
4. Smoke-test the installer and portable EXE on every supported Windows version.
5. Independent security review / penetration test.
6. Accessibility review of the main UI against WCAG 2.1 AA (the UI currently has few ARIA labels).
