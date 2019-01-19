import numpy as np
import cv2


def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = img
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def isIntersect(human, knife):

    amin_x = human[0]
    amax_x = human[2]
    amin_y = human[1]
    amax_y = human[3]

    othermin_x = knife[0]
    othermax_x = knife[2]
    othermin_y = knife[1]
    othermax_y = knife[3]


    if amin_x > othermax_x or amax_x < othermin_x:
        return False
    if amin_y > othermax_y or amax_y < othermin_y:
        return False
    return True
