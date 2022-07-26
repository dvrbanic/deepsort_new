#!/usr/bin/python3

from deepsort import *

import rospy
from uav_object_tracking_msgs.msg import object, objectList
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def get_gt(image, frame_id, gt_dict):
    if frame_id not in gt_dict.keys() or gt_dict[frame_id] == []:
        return None, None, None

    frame_info = gt_dict[frame_id]

    detections = []
    ids = []
    out_scores = []
    for i in range(len(frame_info)):
        coords = frame_info[i]['coords']

        x1, y1, w, h = coords
        x2 = x1 + w
        y2 = y1 + h

        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)

        detections.append([x1, y1, w, h])
        out_scores.append(frame_info[i]['conf'])

    return detections, out_scores


def get_dict(filename):
    with open(filename) as f:
        d = f.readlines()

    d = list(map(lambda x: x.strip(), d))

    last_frame = int(d[-1].split(',')[0])

    gt_dict = {x: [] for x in range(last_frame + 1)}

    for i in range(len(d)):
        a = list(d[i].split(','))
        a = list(map(float, a))

        coords = a[2:6]
        confidence = a[6]
        gt_dict[a[0]].append({'coords': coords, 'conf': confidence})

    return gt_dict


def get_mask(filename):
    mask = cv2.imread(filename, 0)
    mask = mask / 255.0
    return mask


def callback(data):
    global detections
    global out_scores
    global boolean

    detections = []
    out_scores = []

    for i in range(len(data.obj)):
        bb = data.obj[i]

        detections.append([bb.x, bb.y, bb.width, bb.height])
        out_scores.append(bb.confidence)
    rospy.loginfo(detections)
    rospy.loginfo(out_scores)
    boolean = True


def image_callback(data):
    global tracker
    global detections
    global cv_image
    global boolean2

    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

    if tracker is not None:
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            id_num = str(track.track_id)  # Get the ID for the particular track.
            features = track.features  # Get the feature vector corresponding to the detection.

            # Draw bbox from tracker.
            cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(cv_image, str(id_num), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
        for det in detections:
            cv2.rectangle(cv_image, (int(det[0]), int(det[1])), (int(det[0])+int(det[2]), int(det[1])+int(det[3])), (255, 0, 0), 2)


    boolean2 = True
    cv2.imshow("sort", cv_image)
    cv2.waitKey(1)



if __name__ == '__main__':
    global cv_image
    detections = []
    out_scores = []
    boolean = False
    boolean2 = False

    rospy.init_node('sort', anonymous=True)
    subscriber = rospy.Subscriber("/YOLODetection/detected_objects", objectList, callback)
    image_sub = rospy.Subscriber("/zedm/zed_node/rgb/image_rect_color", Image, image_callback)
    pub = rospy.Publisher('/trackers', objectList, queue_size=10)


    r = rospy.Rate(6)
    deepsort = deepsort_rbc(wt_path='/home/dominik/catkin_ws_deep_sort/src/deep_sort/src/model_44.pt') 

    while not rospy.is_shutdown():
        if boolean is True and boolean2 is True and len(detections) != 0:
            detections = np.array(detections)
            out_scores = np.array(out_scores)
            tracker, detections_class = deepsort.run_deep_sort(cv_image, out_scores, detections)

            lista_trackers = objectList()
            for ct, trck in enumerate(tracker.tracks):
                if not trck.is_confirmed() or trck.time_since_update > 1:
                    continue
                bbox = trck.to_tlbr()
                tmp_obj = object()
                tmp_obj.x = (bbox[0]+bbox[2])/2
                tmp_obj.y = (bbox[1]+bbox[3])/2
                tmp_obj.width = bbox[2]-bbox[0]
                tmp_obj.height = bbox[3]-bbox[1]
                #tmp_obj.confidence = out_scores[ct]
                lista_trackers.obj.append(tmp_obj)
            pub.publish(lista_trackers)

            boolean = False
            boolean2 = False
        else:
            tracker = None
            continue
        

        r.sleep()
