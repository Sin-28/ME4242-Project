"""
Live preview for EMIO camera calibration.

`python -m emioapi calibrate` shows blank windows until it accepts a frame, so you can't see
what is wrong. This preview runs the same acceptance checks as emioapi on every frame and
shows why a frame would be rejected:
  - no marker / more than one marker / wrong ID (must be ArUco ORIGINAL id 672)
  - missing depth at any of the 9 points emioapi samples (4 corners, 4 edge midpoints, centre)

Usage (ME4242 env, EMIO Labs closed):
    python calib_preview.py
Position the marker until the status says READY for most frames, press q,
then run `python -m emioapi calibrate` (only one program can use the camera at a time).

Window keys: q / Esc quit, s save a snapshot.
"""

import time
from collections import deque

import cv2
import numpy as np

from emioapi import EmioCamera

MARKER_ID = 672
THRESHOLD = 95                     # emioapi zeroes pixels darker than this before detecting
WINDOW = "Calibration preview  (q = quit, s = snapshot)"
# where the marker corners appear with the factory calibration (emioapi's default camera_2d_points.csv)
EXPECTED_CORNERS = np.array([[217, 294], [216, 178], [304, 187], [305, 284]], np.int32)

GREEN, RED, ORANGE, GREY, WHITE = (0, 200, 0), (0, 0, 255), (0, 165, 255), (160, 160, 160), (255, 255, 255)


def sample_points(corners):
    """The 9 pixels emioapi reads depth at: 4 corners, 4 edge midpoints, centre."""
    pts = [tuple(c) for c in corners]
    pts += [tuple((corners[i] + corners[(i + 1) % 4]) / 2) for i in range(4)]
    pts.append(tuple(corners.mean(axis=0)))
    return [(int(x), int(y)) for x, y in pts]


def check_frame(frame, depth, detector):
    """Returns (accepted, reason, corners_of_672, all_corners, all_ids, rejected, thresholded)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_TOZERO)
    corners, ids, rejected = detector.detectMarkers(thresh)
    if ids is None:
        return False, "no marker found", None, corners, ids, rejected, thresh
    if len(ids) > 1:
        return False, f"{len(ids)} markers found (ids {ids.flatten().tolist()}), need exactly one", \
            None, corners, ids, rejected, thresh
    if ids[0][0] != MARKER_ID:
        return False, f"wrong marker id {ids[0][0]} (need {MARKER_ID})", None, corners, ids, rejected, thresh
    c = corners[0][0]
    missing = [p for p in sample_points(c)
               if not (0 <= p[1] < depth.shape[0] and 0 <= p[0] < depth.shape[1]) or depth[p[1], p[0]] == 0]
    if missing:
        return False, f"no depth at {len(missing)} of 9 sample points", c, corners, ids, rejected, thresh
    return True, "READY: emioapi would accept this frame", c, corners, ids, rejected, thresh


def draw(frame, depth, result, history):
    accepted, reason, c672, corners, ids, rejected, thresh = result
    out = frame.copy()
    cv2.polylines(out, [EXPECTED_CORNERS], True, GREY, 1, cv2.LINE_AA)
    cv2.putText(out, "factory position", (EXPECTED_CORNERS[1][0] - 20, EXPECTED_CORNERS[1][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)
    for r in rejected:                                   # square-ish shapes that failed decoding
        cv2.polylines(out, [r[0].astype(np.int32)], True, ORANGE, 1)
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            colour = GREEN if i == MARKER_ID else RED
            cv2.polylines(out, [c[0].astype(np.int32)], True, colour, 2)
            cv2.putText(out, f"id {i}", tuple(c[0][0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    if c672 is not None:
        for k, p in enumerate(sample_points(c672)):
            ok = 0 <= p[1] < depth.shape[0] and 0 <= p[0] < depth.shape[1] and depth[p[1], p[0]] > 0
            cv2.circle(out, p, 4, GREEN if ok else RED, -1)
        top = ((c672[0] + c672[1]) / 2).astype(int)   # marker's top edge: should face the camera
        cv2.putText(out, "top", tuple(top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

    rate = 100 * sum(history) / max(len(history), 1)
    lines = [reason, f"accepted {rate:.0f}% of the last {len(history)} frames",
             "green = id 672 | red = other id | orange = rejected square | dots = depth samples"]
    for k, text in enumerate(lines):
        y = 22 + 20 * k
        colour = GREEN if (k == 0 and accepted) else RED if k == 0 else WHITE
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    # right panel: what the detector actually sees (after emioapi's darkness threshold)
    panel = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel, "what the detector sees", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)
    return np.hstack([out, panel])


def main():
    camera = EmioCamera(show=False, track_markers=False, compute_point_cloud=False)
    if not camera.open():
        raise SystemExit("Could not open the camera. Close EMIO Labs and any other script using it.")
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL),
                                       cv2.aruco.DetectorParameters())
    history = deque(maxlen=60)
    last_reason = None
    print("Showing live preview. Press q in the window to quit.")
    try:
        while True:
            camera.update()
            frame, depth = camera.frame.copy(), camera.depth_frame.copy()
            result = check_frame(frame, depth, detector)
            history.append(result[0])
            if result[1] != last_reason:                 # print only when the status changes
                print(result[1])
                last_reason = result[1]
            view = draw(frame, depth, result, history)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = f"calib_preview_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(name, view)
                print(f"Saved {name}")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
