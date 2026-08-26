# Build verification notes

Validation performed while assembling this prototype:

- Python source tree: `python -m compileall` completed successfully after the final code patches.
- Native QKP reference solver: CMake Release build completed successfully and its unit test passed.
- Python exact QKP fallback: compared against brute force on 20 random 8-variable instances with positive and negative pair terms; all objectives matched.
- B3 COTAHIST parser: fixed-width parser has a synthetic layout-conformance unit test using the official field positions.
- ML evaluation: unit test verifies Rank IC is computed inside each prediction-date cross-section.

The artifact sandbox did not have the project's external Python/npm dependencies available and did not allow package installation from public registries during the final validation pass. Therefore the full pytest suite, Next.js build, and Docker Compose end-to-end startup are delegated to the included CI/container build on a networked development machine.

This limitation is about build-time dependency availability, not an intentional stub: Dockerfiles and CI install the declared dependencies and execute those checks.
