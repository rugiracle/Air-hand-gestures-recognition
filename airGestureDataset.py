import cv2
import numpy as np
from torchvision import transforms


class AirGestureDataset():
    """Prepares gestures written in the air to look like MNIST digit dataset"""
    def __init__(self, path, color=(1, 1, 1), offset=40, fillin=0, hCam=480, wCam=640):
        self.path = path       # path to gesture db
        self.color = color     # color of gesture
        self.offset = offset    # to mimic Mnist data set
        self.fillin = fillin    # background color
        self.hCam = hCam        # camera image height
        self.wCam = wCam        # camera image width
        self.display_img = np.ones((self.hCam, self.wCam)) * self.fillin # drawing image
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))])
        self. drop_n_pt = 6  # for additional gesture spotting we drop n points from both hand trajectory ends

    def draw_trajectory(self, htrajector):
        """Draw gesture trajectory on a black image and crop the gesture with a margin of size offset
        the cropped image will be recognized by a pretrained network on MNIST digit dataset"""
        length = len(htrajector)
        array_pt = np.array(htrajector)
        min_x = min(array_pt[:, 0])
        min_y = min(array_pt[:, 1])
        max_x = max(array_pt[:, 0])
        max_y = max(array_pt[:, 1])

        self.display_img.fill(self.fillin)  # reset the drawing image
        for i in range(length - self.drop_n_pt-1, self.drop_n_pt, -1):  # drop a few points on both ends of the gesture
            cv2.line(self.display_img, htrajector[i - 1], htrajector[i], self.color, 20)
        crop = self.display_img[min_y - self.offset:max_y + self.offset * 2, min_x - self.offset: max_x + self.offset * 2]
        crop_tr = self.transform(crop)
        #cv2.imshow('cropped', crop)
        #cv2.waitKey(1000)
        return crop_tr

    def getHandGesture(self):
        """from a text files, the function creates air handwritten gestures images of the size 28x28"""
        self.display_img = np.ones((self.hCam, self.wCam)) * self.fillin
        gestures = []
        with open(self.path, 'r') as fp:
            for line in fp:
                if len(line) > 100:
                    one_gesture = []
                    list_pts = line.split(' ')
                    list_pts = list_pts[:len(list_pts) - 1]
                    for pts in list_pts:
                        pt = pts.split(',')
                        x, y = int(pt[0]), int(pt[1])
                        one_gesture.append([x, y])
                    air_gesture = self.draw_trajectory(one_gesture)
                    gestures.append(air_gesture)
        return gestures


