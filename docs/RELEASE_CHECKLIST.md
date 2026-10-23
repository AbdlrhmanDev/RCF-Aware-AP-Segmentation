# Public code-release checklist

- [ ] Open the source code and confirm no API keys, passwords, user credentials, emails, patient identifiers, or restricted image paths are present.
- [ ] Confirm the source files are the intended final experimental versions. `C2/seed_2026` is sourced from the later recovery execution cell, not an earlier incomplete attempt.
- [ ] Confirm that all notebook output cells are empty (already removed here).
- [ ] Ensure `.gitignore` is applied and no raw PRAD images, masks, checkpoints, manifests with sensitive identifiers, screenshots, or Drive tokens are committed.
- [ ] Choose an open-source license only with the authorization of all code rights holders; replace `LICENSE_DECISION.md` with a real LICENSE if appropriate.
- [ ] Decide whether to release (a) source code alone, (b) authorized synthetic/demo data, or (c) private data-access instructions. Do not invent or reconstruct historical outputs.
- [ ] If allowed, provide original, verifiable frozen protocol JSON, non-sensitive split metadata, and manifest schema as separate versioned releases. Do not publish a synthetic replacement claiming the historical hash.
- [ ] Verify the journal's requirements for data/code availability, anonymized submission, license, and persistent links (e.g. GitHub release archived with Zenodo).
- [ ] Publish the repository, tag a release, and update Code Availability with its real public URL and archived DOI if available.
- [ ] Recheck permission and de-identification before publishing any example radiographs; no radiographs are in this bundle.
- [ ] Do not claim an independently confirmed end-to-end rerun of released code: the build verified syntax and provenance, not scientific re-execution.
