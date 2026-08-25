# WAKE_README.md — how to bring this box back

The programme finished and the box was suspended by `scripts/finalize_and_suspend.sh`.
Everything is committed, pushed, and uploaded to HuggingFace. Nothing is mid-flight.

Two self-starting things were deliberately disabled so that "suspended" meant suspended:

```bash
# re-arm the decode-loop watchdog (restarts the decode loop if it dies)
systemctl --user start dsv4-decode-loop-watchdog.timer

# re-arm the hourly flywheel agent
rm FLYWHEEL_STOP
```

State when it went to sleep is in `FINAL_STATE.md`; the archive hash pass is in
`evidence/archive_verify_final.txt`.
