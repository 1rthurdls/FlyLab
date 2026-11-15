"""Edit the real physics videos of the extras into finished clips: title card + segments with a caption bar.

Usage: .venv/Scripts/python.exe scripts/make_clips.py [clip names]   -> runs/videos/<clip>.mp4 (1280x720, 30 fps)
Captions quote the numbers of the actual runs (result.json). Silent videos get a silent audio track so clips
can be concatenated and played anywhere.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
OUT = RUNS / "videos"
FF = imageio_ffmpeg.get_ffmpeg_exe()
FONT = "C\\:/Windows/Fonts/segoeui.ttf"
FONT_B = "C\\:/Windows/Fonts/segoeuib.ttf"
W, H = 1280, 720


def res(run_id):
    return json.loads((RUNS / run_id / "result.json").read_text(encoding="utf8"))


def _txt(tmp, name, text):
    p = Path(tmp) / f"{name}.txt"
    p.write_text(text, encoding="utf8")
    return str(p).replace("\\", "/").replace(":", "\\:")


def _run(args):
    r = subprocess.run([FF, "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])


def _dur(path):
    return imageio_ffmpeg.count_frames_and_secs(str(path))[1]


def _has_audio(path):
    r = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True)
    return "Audio:" in r.stderr


def title_card(tmp, idx, title, lines, secs=3.5):
    out = Path(tmp) / f"{idx:02d}_title.mp4"
    tf, lf = _txt(tmp, f"t{idx}", title), _txt(tmp, f"l{idx}", "\n".join(lines))
    vf = (f"drawtext=fontfile='{FONT_B}':textfile='{tf}':fontcolor=0xF0EFEC:fontsize=54:x=(w-text_w)/2:y=h/2-120,"
          f"drawtext=fontfile='{FONT}':textfile='{lf}':fontcolor=0xC3C2B7:fontsize=28:line_spacing=12:"
          f"x=(w-text_w)/2:y=h/2-20,"
          f"drawtext=fontfile='{FONT}':text='FlyLab · whole-brain fly model (FlyWire) + physics body':"
          f"fontcolor=0x8A897F:fontsize=22:x=(w-text_w)/2:y=h-70")
    _run(["-f", "lavfi", "-i", f"color=c=0x141413:s={W}x{H}:d={secs}:r=30", "-f", "lavfi", "-t", str(secs),
          "-i", "anullsrc=r=44100:cl=stereo", "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
          "-crf", "20", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)])
    return out


def segment(tmp, idx, src, caption, max_secs=None):
    out = Path(tmp) / f"{idx:02d}_seg.mp4"
    cf = _txt(tmp, f"c{idx}", caption)
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x141413,fps=30,"
          f"drawbox=x=0:y=ih-96:w=iw:h=96:color=0x141413@0.78:t=fill,"
          f"drawtext=fontfile='{FONT}':textfile='{cf}':fontcolor=0xF0EFEC:fontsize=25:line_spacing=8:x=28:y=h-84")
    dur = _dur(src) if max_secs is None else min(_dur(src), max_secs)
    args = ["-i", str(src)]
    if not _has_audio(src):
        args += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
        amap = ["-map", "0:v", "-map", "1:a"]
    else:
        amap = ["-map", "0:v", "-map", "0:a"]
    _run([*args, *amap, "-t", f"{dur:.3f}", "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
          "-crf", "20", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", str(out)])
    return out


def build(name, title, lines, segs):
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts = [title_card(tmp, 0, title, lines)]
        for k, (src, cap) in enumerate(segs, 1):
            parts.append(segment(tmp, k, RUNS / src, cap))
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{str(p).replace(chr(92), '/')}'\n" for p in parts), encoding="utf8")
        out = OUT / f"{name}.mp4"
        _run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    print(f"{out}  ({_dur(out):.0f} s, {out.stat().st_size / 1e6:.1f} MB)", flush=True)


def clips():
    sw, sp = res("beh_swat-normal_232dc661"), res("beh_swat-normal-silence-giant-fiber_ea44838e")
    bk, bl = res("beh_bike-loop_f235fa1a"), res("beh_bike-loop-silence-DNa01-left-DNa02-left_a85c3b93")
    pk = res("beh_piano-keyboard_fee2db92")
    q, o = res("beh_song-road_159b7148"), res("beh_song-road_897ce716")
    left = lambda r: next(s for s in r["segments"] if s["action"] == "left")  # noqa: E731
    return {
        "01_fly_swatter": ("Fly swatter", ["A swatter comes down on a walking fly.",
                                           "Its looming image drives the looming neurons LPLC2 + LC4 -> giant fiber.",
                                           "Then we cut the giant fiber.  (10x slow motion; take-off is scripted)"], [
            ("beh_swat-normal_232dc661/video.mp4",
             f"INTACT BRAIN: giant fiber fired with the swatter still {sw['swatter_distance_at_gf_mm']:.0f} mm away"
             f"\noff the ground {sw['margin_ms']} ms before impact -> ESCAPE"),
            ("beh_swat-normal-silence-giant-fiber_ea44838e/video.mp4",
             "GIANT FIBER SILENCED: it never fires (peak 0 Hz)\nno escape -> SPLAT"),
        ]),
        "02_bicycle": ("The fly rides a bike", ["The bike reads the fly's real LEGS (physics simulation),",
                                                "like the floating ball a tethered fly walks on in the lab.",
                                                "Then we cut the left steering neurons.  (4x slow motion)"], [
            ("beh_bike-loop_f235fa1a/video.mp4",
             f"INTACT: 'turn left' = P9 + left DNa01/DNa02 -> legs turn {left(bk)['ball']['fly_yaw_deg_s']:+.0f} deg/s"
             f"\nthe bike turns {left(bk)['bike']['heading_change_deg']:+.0f} deg"),
            ("beh_bike-loop-silence-DNa01-left-DNa02-left_a85c3b93/video.mp4",
             f"LEFT DNa01 + DNa02 SILENCED: the same 'turn left' -> legs {left(bl)['ball']['fly_yaw_deg_s']:+.0f} deg/s"
             f"\nthe bike turns {left(bl)['bike']['heading_change_deg']:+.0f} deg (it can't turn left)"),
        ]),
        "03_fly_piano": ("Fly piano", ["Three flies walk across a giant keyboard (each its own brain + body).",
                                       "Every time a front foot touches down, the key under it plays.",
                                       "Rhythm = their real footsteps.  (8x slow motion, sound on)"], [
            ("beh_piano-keyboard_fee2db92/video.mp4",
             f"fly 1 walking forward (shown) + fly 2 turning + fly 3 walking backward"
             f"\n{pk['n_notes']} notes from footsteps in 1.5 s of walking"),
        ]),
        "04_22_questions_walking_fly": ("22 Questions, played by a walking fly",
                                        ["Your arrangement's keys are laid on the floor just ahead of the fly.",
                                         "A key sounds the first time a foot lands on it.",
                                         "The melody is the score; the rhythm is the fly's own steps.  (8x slow motion)"], [
            ("beh_song-road_159b7148/video.mp4",
             f"P9 (walking command) drives the legs; {q['notes_played']}/{q['notes_total']} notes played"
             f"\nwalked {q['walked_mm']:.0f} mm; each key sounds on the first footstep"),
        ]),
        "05_ode_to_joy_walking_fly": ("Ode to Joy, played by a walking fly",
                                      ["Same musical road with a public-domain tune.",
                                       "A stumble can swap two notes or skip one.  (8x slow motion)"], [
            ("beh_song-road_897ce716/video.mp4",
             f"{o['notes_played']}/{o['notes_total']} notes, ~{o['tempo_bpm_heard']:.0f} BPM set by the fly's stepping"
             f"\n{o['notes_out_of_order']} notes swapped by a stumble"),
        ]),
    }


if __name__ == "__main__":
    allc = clips()
    for name in (sys.argv[1:] or list(allc)):
        title, lines, segs = allc[name]
        build(name, title, lines, segs)
