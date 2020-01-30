import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from core.detectors import CornerNet_Squeeze
from core.vis_utils import draw_bboxes

def cam(detector):
    cap = cv2.VideoCapture(0)   
    cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # カメラ画像の横幅を1280に設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # カメラ画像の縦幅を720に設定

    while True:
        # VideoCaptureから1フレーム読み込む
        ret, frame = cap.read()
        
        image, bboxes = obj_inference(detector, frame)

        # 加工なし画像を表示する
        cv2.imshow('Raw Frame', image)
        
        # キー入力を1ms待って、k が27（ESC）だったらBreakする
        k = cv2.waitKey(1)
        if k == 27:
            break

    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    cv2.destroyAllWindows()

def obj_inference(detector, image):   
    bboxes = detector(image)
    #print(bboxes)
    image  = draw_bboxes(image, bboxes)
    #cv2.imwrite("demo_out.jpg", image)
    return image, bboxes

def main():
    detector = CornerNet_Squeeze()
    cam(detector)

if __name__ == "__main__":
    main()