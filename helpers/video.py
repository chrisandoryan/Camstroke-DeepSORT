import cv2

def get_video_size(video_path):
    vid = cv2.VideoCapture(video_path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    return (width, height)

def frame_to_video(frames, output_path, w, h):
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

def get_fps(video_path):
    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    return fps
    