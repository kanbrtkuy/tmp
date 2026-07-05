# Fable Review Handoff Protocol

Date: 2026-07-05

Use this protocol whenever a Stage1/Stage2 review packet needs Fable/external review.

## Default Path

1. Prepare a content-quiet review packet:
   - aggregate counts only
   - artifact paths and hashes
   - no raw prompts
   - no raw CoT/completions
   - no unreleased example text
2. Commit/push the packet to the GitHub tmp review repo.
3. Ask Fable to review the GitHub tmp repo link/path.
4. Save Fable's response back into the repo as a review artifact.

## If Direct Fable Channel Is Blocked

Do not keep retrying the blocked direct-send path.

Instead:

1. Push the sanitized packet to the GitHub tmp repo.
2. Give Fable only the GitHub tmp repo link/path.
3. Continue local audit work while waiting.
4. Record that direct handoff was blocked and that GitHub tmp handoff was used.

## Required Review Packet Metadata

Every packet should state:

- date
- code commit
- RunPod artifact root
- exact data freeze directory
- exact audit outputs
- known blockers
- reviewer questions
- whether human QA has passed

