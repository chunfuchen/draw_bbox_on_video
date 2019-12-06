import skvideo.io
import argparse
import json
import cv2

parser = argparse.ArgumentParser(description='Plot bounding box on video')

# model definition
parser.add_argument('path', type=str, metavar='PATH',
                    help='path to json')

args = parser.parse_args()

class BBoxInfo:
    def __init__(self, name, bbox, video_info, confidence):

        self.name = name
        self.top_left = (int(bbox['Left'] * video_info['width']), int(bbox['Top'] * video_info['height']))
        self.bottom_right = (int((bbox['Left'] + bbox['Width']) * video_info['width']),
                             int((bbox['Top'] + bbox['Height']) * video_info['height']))

        self.confidence = confidence

        self.color = 'red'

        self.text_top_left = (self.top_left[0], max(0, self.top_left[1] - 2))

    def draw_self(self, img):
        # TODO: add color, font and size
        cv2.rectangle(img, self.top_left, self.bottom_right, color=(255, 0, 0), thickness=2)
        cv2.putText(img, self.name + " {:4.2f}".format(self.confidence), self.text_top_left,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
        return img

    def __str__(self):
        return "{}: ({}, {})".format(self.name, self.top_left, self.bottom_right)

def main():
    with open(args.path) as f:
        detection_results = json.load(f)

    video_name = detection_results['fileName']
    video_metadata = skvideo.io.ffprobe(video_name)['video']
    video_info = {'width': float(video_metadata['@width']), 'height': float(video_metadata['@height']),
                  'fps': eval(video_metadata['@avg_frame_rate'])}
    video = skvideo.io.vread(video_name)

    all_bbox_info = {}

    for v in detection_results['Labels']:
        timestamp = v['Timestamp']
        name = v['Label']['Name']
        for ins in v['Label']['Instances']:
            if timestamp not in all_bbox_info:
                all_bbox_info[timestamp] = []
            all_bbox_info[timestamp].append(BBoxInfo(name, ins['BoundingBox'], video_info, ins['Confidence']))

    for t, bbox_infos in all_bbox_info.items():
        if int(t / video_info['fps']) > video.shape[0]:
            continue
        tmp = video[int(t / video_info['fps']), ...]
        for tt in bbox_infos:
            tmp = tt.draw_self(tmp)

        video[int(t / video_info['fps']), ...] = tmp


    skvideo.io.vwrite(video_name[:-4] + "_bbox.mp4", video)

if __name__ == "__main__":
    main()
