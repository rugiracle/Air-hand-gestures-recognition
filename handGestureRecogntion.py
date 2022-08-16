import handDetector as hd
import mnistDigitModel as gr
import torch
import airGestureDataset as ag
import matplotlib.pyplot as plt
import cv2
import numpy as np

N_CLASSES = 10


def transfer_learning_airDigit(device, use_cuda, path_model,  path_gesture):
    """
    Transfer learning:
    Use a pretrained model on Mnist digit dataset[the path to the model is given: path_model]
     to recognize air handwritten digits
     the path to the air handwritten gestures is given as well: path_gesture
    """
    model = gr.MnistNet(n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(path_model))
    model.eval()
    print('*************  Transfer learning of the model @' + path_model + '  *************')

    gestures_samples = ag.AirGestureDataset(path_gesture)
    gestures = gestures_samples.getHandGesture()
    ROW_IMG = int(len(gestures) / 2)
    N_ROWS = 2
    fig = plt.figure()
    for i in range(1, ROW_IMG * N_ROWS + 1):
        test_img1 = gestures[i - 1]
        plt.subplot(N_ROWS, ROW_IMG, i)
        plt.axis('off')
        plt.imshow(test_img1.permute((1, 2, 0)), cmap='gray_r')
        title = f'{gr.predict_image(test_img1, device, use_cuda, model)} '
        plt.title(title, fontsize=7, color='b')
    fig.suptitle('Air Gestures- predictions')
    plt.savefig('model/AirGestures_predictions.png')
    plt.show()


def draw_trajectory(img, index_position, drop):
    length = len(index_position)
    for i in range(length-1-drop, drop, -1):
        cv2.line(img, index_position[i - 1], index_position[i], (0, 0, 0), 5)


def online_gesture_detection(device, use_cuda, path_model, hCam, wCam, path_gesture):
    """
    Transfer learning:
    Use a pretrained model on Mnist digit dataset[the path to the model is given: path_model]
     to recognize air handwritten digits
     the path to the air handwritten gestures is given as well: path_gesture
    """
    model = gr.MnistNet(n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(path_model))
    model.eval()
    print('*************  Transfer learning of the model @' + path_model + '  *************')
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = hd.HandDetector(detectionConf=0.7)
    hand_positions = []
    gestures = ag.AirGestureDataset(path_gesture)
    display_img = np.ones((hCam, wCam)) * 255

    title = "  "
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)
        if len(lm_list):
            bbox = detector.hand_bbox(lm_list)
            cv2.rectangle(img, bbox[0], bbox[1], (0, 222, 0), 3)
            w = bbox[1][0] - bbox[0][0]
            h = bbox[1][1] - bbox[0][1]
            cx, cy = bbox[0][0] + int(w / 2), bbox[0][1] + int(h / 2)
            cv2.circle(img, (cx, cy), 10, (200, 0, 0), cv2.FILLED)
            hand_positions.append((cx, cy))
            if len(hand_positions) > 30:
                draw_trajectory(img, hand_positions, 6)
                display_img.fill(255)
                draw_trajectory(display_img, hand_positions, 0)
                """ helps for gesture spotting: end of a gesture """
                if lm_list[8][1] - lm_list[7][1] > 0 and lm_list[8][2] - lm_list[7][2] > 0:
                    gesture_img = gestures.draw_trajectory(hand_positions)
                    hand_positions = []
                    plt.axis('off')
                    plt.imshow(gesture_img.permute((1, 2, 0)), cmap='gray_r')
                    title = f'{gr.predict_image(gesture_img, device, use_cuda, model)} '
                    plt.title(title, fontsize=7, color='b')
                    cv2.putText(img, title, (450, 140), cv2.FONT_HERSHEY_SIMPLEX, 5, (200, 0, 0), 10)

        else:
            display_img.fill(255)
            hand_positions = []
        cv2.putText(display_img, title, (450, 120), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)
        cv2.imshow("frame", img)
        cv2.imshow("Display", display_img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    PATH = 'model/mnist_50_0.55_128.pt'
    #gr.testing_model(batch_size=1000, device=device, PATH=PATH)
    path_gesture = 'gestures.txt'
    #transfer_learning_airDigit(device, use_cuda, PATH, path_gesture)
    width_cam, height_cam = 640, 480
    online_gesture_detection(device, use_cuda, PATH, height_cam, width_cam, path_gesture)


if __name__ == '__main__':
    main()