# Camera watcher - deployment

Cron, single pass per run (lock makes overlapping runs safe):

    */10 * * * * cd /home/Sebastian-Gerken/OTVision && \
      .venv/bin/python watch_cameras.py --once \
      "/Volumes/platomo data/Projekte/OTC015_Team-Red/videos" \
      >> logs_track/watch.log 2>&1

Knobs: --block-days 4, --idle-minutes 5, --stable-minutes 5, --slots-per-day 96,
--reserve-cores 2, --max-parallel 1, --cores-per-track 4, --max-failures 3.

Per-camera state (in the camera dir):
- .otc_watch_state.json -> tracked_through (downstream ready signal)
- .otc_watch_scan.json -> stability snapshot
- lock: .locks/<camera>-<hash>.lock under the repo
- host-wide track slots: .locks/slots/slot-*.lock bound total track.py processes
  across overlapping cron runs

Recovery: crash never advances tracked_through, so the block retries next poll.
To reprocess a camera, delete .otc_watch_state.json. flock releases
automatically when a process dies; no stale locks to clear.
Repeated track/verify failures back off in .otc_watch_state.json, then quarantine
after --max-failures; delete or repair that marker entry after manual recovery.

Downstream MUST gate on tracked_through, not raw .ottrk mtime (track writes
.ottrk non-atomically). Leftover complete days < block-days wait for a full
block; to flush a tail, run once with --block-days 1.
