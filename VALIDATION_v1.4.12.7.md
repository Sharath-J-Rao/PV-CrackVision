# Validation v1.4.12.7

- `npm run lint` (tsc --noEmit), `npm test` (server tests incl. `server/v14127.test.mjs`), `python -m unittest discover -s training` and `npm run build` were run on this source.
- Not exercised here: real GPU training, DirectML inference on a Windows GPU, and the effect of the new filters on real orthomosaic images. Tune *Fragment overlap*, *Reject below confidence* and the size gates in Settings if boxes are still over- or under-filtered.
