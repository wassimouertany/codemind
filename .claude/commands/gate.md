---
description: Verify the current build-order gate before moving to the next phase
---
Read CLAUDE.md, identify which build-order phase we are currently in, then:
1. Run `make lint` and `make test`. Report failures without fixing them yet.
2. State the gate criterion for the current phase, in one sentence.
3. Produce evidence that the gate passes (a test, a number, a command output).
   If you cannot produce evidence, say so plainly and stop.
Do not write new feature code in this command.
