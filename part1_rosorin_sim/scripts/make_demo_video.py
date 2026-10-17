#!/usr/bin/env python3
"""Cut the three stage recordings (RECORD=1) into one 3-5 minute Part 1 demo video.

  media/01_teleop_sensors.mp4  -> run & control the robot, sensor visualisation (real time)
  media/02_slam_mapping.mp4    -> SLAM + autonomous exploration (sped up), then the saved map
  media/03_navigation.mp4      -> single goal, then multi-point navigation (sped up)

Captions are placed from the wall-clock timestamps of the ROS log lines in logs/teleop_demo.log and
logs/nav_demo.log, aligned on the end of each recording. Runs on the host, needs ffmpeg.

  python3 scripts/make_demo_video.py [--slam 100] [--single 45] [--multi 90] [--out media/part1_demo.mp4]
"""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MEDIA, LOGS, MAPS = HERE / "media", HERE / "logs", HERE / "maps"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W, H, FPS = 1280, 720, 30
ROS_LINE = re.compile(r"\[INFO\] \[(\d+\.\d+)\] \[[^\]]+\]: (.*)")


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def log_events(path, video_s, tail_s):
    """[(video time, message)] aligned on the END of the recording: stage.sh stops ffmpeg `tail_s` after
    the demo script's last log line, while the start offset varies (record_start waits for the encoder)"""
    ev = [(float(m.group(1)), m.group(2)) for m in map(ROS_LINE.search, path.read_text().splitlines()) if m]
    return [(video_s - tail_s - (ev[-1][0] - t), msg) for t, msg in ev] if ev else []


class Cutter:
    def __init__(self, tmp):
        self.tmp, self.parts = Path(tmp), []

    def _text(self, s):
        f = self.tmp / f"txt{len(list(self.tmp.glob('txt*')))}.txt"
        f.write_text(s)
        return f

    def caption(self, s, y=24, size=30, enable=None):
        en = f":enable='between(t,{enable[0]:.2f},{enable[1]:.2f})'" if enable else ""
        return (f"drawtext=fontfile={FONT}:textfile={self._text(s)}:fontsize={size}:fontcolor=white"
                f":box=1:boxcolor=black@0.65:boxborderw=12:x=24:y={y}{en}")

    def clip(self, src, start, end, speed, title, sub_captions=()):
        """src[start:end] played `speed` x faster with a title; sub_captions = [(t0, t1, text)] in source time."""
        vf = [f"trim={start:.2f}:{end:.2f}", "setpts=PTS-STARTPTS", f"setpts=PTS/{speed:.4f}",
              f"scale={W}:{H}", f"fps={FPS}", self.caption(title)]
        for t0, t1, text in sub_captions:  # source time -> output time
            a, b = (t0 - start) / speed, (t1 - start) / speed
            if b > 0:
                vf.append(self.caption(text, y=H - 80, size=28, enable=(max(a, 0), b)))
        self._render(["-i", str(src)], ",".join(vf))

    def card(self, lines, seconds=4.0, image=None):
        inputs = ["-f", "lavfi", "-i", f"color=c=0x101820:s={W}x{H}:d={seconds}:r={FPS}"]
        y = 40 if image else H // 2 - 40 * len(lines)
        texts = []
        for i, s in enumerate(lines):
            size = 44 if i == 0 else 30
            texts.append(f"drawtext=fontfile={FONT}:textfile={self._text(s)}:fontsize={size}:fontcolor=white"
                         f":x=(w-text_w)/2:y={y}")
            y += size + 22
        if image:  # map image centred under the text
            inputs += ["-loop", "1", "-t", str(seconds), "-i", str(image)]
            graph = (f"[1:v]scale=-2:{H - 200}:flags=neighbor[img];[0:v][img]overlay=(W-w)/2:170:shortest=1,"
                     + ",".join(texts))
            self._render(inputs, graph, complex_=True)
        else:
            self._render(inputs, ",".join(texts))

    def _render(self, inputs, vf, complex_=False):
        out = self.tmp / f"part{len(self.parts):02d}.mp4"
        flt = ["-filter_complex", vf] if complex_ else ["-vf", vf]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, *flt, "-an", "-c:v", "libx264", "-threads", "4", "-preset",
                        "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS), str(out)], check=True)
        self.parts.append(out)

    def concat(self, out):
        lst = self.tmp / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in self.parts))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-movflags", "+faststart", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slam", type=float, default=100.0, help="target seconds for the SLAM part")
    ap.add_argument("--single", type=float, default=45.0, help="target seconds for single-goal navigation")
    ap.add_argument("--multi", type=float, default=90.0, help="target seconds for multi-point navigation")
    ap.add_argument("--out", default=str(MEDIA / "part1_demo.mp4"))
    args = ap.parse_args()
    teleop, slam, nav = MEDIA / "01_teleop_sensors.mp4", MEDIA / "02_slam_mapping.mp4", MEDIA / "03_navigation.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        c = Cutter(tmp)
        c.card(["ROSOrin Pro - Gazebo simulation", "ROS 2 Humble + Ignition Gazebo Fortress (official Hiwonder package)",
                "Left: Gazebo   |   Right: RViz"], 5)

        # 1-2: teleop. Each segment line is logged when the segment ends; segment lengths from teleop_demo.py
        seg_len = {"forward": 3, "rotate left": 3, "strafe left (mecanum)": 3, "strafe right (mecanum)": 3,
                   "diagonal": 3, "arc": 4, "backward": 3, "stop": 1}
        subs = []
        for t, msg in log_events(LOGS / "teleop_demo.log", duration(teleop), 0.5):
            name = next((k for k in seg_len if msg.startswith(k)), None)
            if name:
                subs.append((t - seg_len[name], t, f"teleop: {name}"))
        c.clip(teleop, 0, duration(teleop), 1.0, "1-2. Teleop + sensors: LiDAR, point cloud, depth cam, odom", subs)

        # 3: SLAM, sped up to the target length
        d = duration(slam)
        k = max(1.0, d / args.slam)
        c.clip(slam, 0, d, k, f"3. SLAM (slam_toolbox) + frontier exploration  x{k:.0f}")

        # 4: saved map
        pgm = MAPS / "rosorin_house.pgm"
        if pgm.exists():
            png = Path(tmp) / "map.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(pgm), str(png)], check=True)
            c.card(["4. Map saved", "ros2 run nav2_map_server map_saver_cli -f rosorin_house"], 6, image=png)

        # 5-6: navigation, split where nav_demo.py starts the waypoint run
        d = duration(nav)
        ev = log_events(LOGS / "nav_demo.log", d, 3.0)
        t_multi = next((t for t, m in ev if m.startswith("multi-point navigation")), d / 3)
        # skip the Nav2 bring-up (RViz shows nothing until AMCL's TF exists): start shortly before the goal
        t_init = max(0.0, next((t for t, m in ev if m.startswith("single-goal navigation")), 0.0) - 8.0)
        k1 = max(1.0, (t_multi - t_init) / args.single)
        c.clip(nav, t_init, t_multi, k1, f"5. AMCL + Nav2: initial pose, single goal  x{k1:.0f}")
        k2 = max(1.0, (d - t_multi) / args.multi)
        c.clip(nav, t_multi, d, k2, f"6. Multi-point navigation (5 waypoints)  x{k2:.0f}")

        c.card(["Results", "metrics in part1_rosorin_sim/results/"], 4)
        c.concat(args.out)
    print(f"wrote {args.out} ({duration(args.out):.0f} s)")


if __name__ == "__main__":
    main()
