# Video: an extension point, not a feature

Nothing in this repo generates video today, and nothing about video has been tested. `media_pipeline/providers/video.py`
fixes the shape a video backend needs so it can reuse the same cost cap, retries and manifest:

- `estimate_cost(request)`: the price of one clip, known before the call. Video APIs usually charge per second of output,
  so the cap check has to happen before submitting, exactly as it does for images.
- `submit(request) -> job_id`: video generation is asynchronous on every major API; you start a job and get an ID back.
- `poll(job_id) -> VideoResult | None`: returns the finished clip, or `None` while it is still rendering. A failed job
  raises a `ProviderError`, so the same retry rules apply.

What a real integration still needs, beyond writing the subclass:

1. A polling loop with a timeout, written into the manifest after every poll so a crash leaves a record.
2. QA on video: sample frames (for example one per second with ffmpeg) and send them to Claude as images, plus a check that
   the clip length and aspect ratio match the brief.
3. A provider account and API key. None was available when this was built, so the interface is untested.
