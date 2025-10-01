import os, glob, json, re
import webvtt

def hms_to_seconds(t):
    hh, mm, ss = t.split(':')
    return int(hh)*3600 + int(mm)*60 + int(float(ss))

out_path = "data/processed/segments.jsonl"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8") as out:
    for vtt_path in glob.glob("data/raw/*.en.vtt"):
        base = os.path.basename(vtt_path)
        # Extract the YouTube ID inside square brackets [ID]
        match = re.search(r"\[([A-Za-z0-9_-]{6,})\]", base)
        video_id = match.group(1) if match else base

        for cue in webvtt.read(vtt_path):
            start_s = hms_to_seconds(cue.start)
            end_s   = hms_to_seconds(cue.end)
            text = re.sub(r"\s+", " ", cue.text).strip()

            # Skip empty lines, [Music], and very short text
            if not text or "[music]" in text.lower() or len(text.split()) < 4:
                continue

            url = f"https://www.youtube.com/watch?v={video_id}&t={int(start_s)}s"
            rec = {
                "video_id": video_id,
                "start": start_s,
                "end": end_s,
                "text": text,
                "url": url
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Wrote:", out_path)
