import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from core.detectors import CornerNet_Squeeze
from core.vis_utils import draw_bboxes, extract_specific_object
import sys

def cam(arg, detector):
    if arg == "video":
        #cap = cv2.VideoCapture('/home/gisen/Documents/rosbag/2019-07-09-15-25-21.avi')
        cap = cv2.VideoCapture('/home/gisen/Documents/rosbag/out_short.mp4')
        width = int(cap.get(3))
        height = int(cap.get(4))
        writer = record(width, height)
    elif arg == "camera":
        cap = cv2.VideoCapture(0)   
        cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # カメラ画像の横幅を1280に設定
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # カメラ画像の縦幅を720に設定

    #while True:
    while (cap.isOpened()):
        # VideoCaptureから1フレーム読み込む
        try:
            ret, frame = cap.read()
            image, bboxes = obj_inference(detector, frame)
        except:
            break

        if arg == "video":
            writer.write(image) # 画像を1フレーム分として書き込み

        # 加工なし画像を表示する
        cv2.imshow('Raw Frame', image)

        # キー入力を1ms待って、k が27（ESC）だったらBreakする
        k = cv2.waitKey(1)
        if k == 27:
            break

    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

def obj_inference(detector, image):   
    bboxes = detector(image)
    #print(bboxes)
    extract_specific_object(image, bboxes)
    image  = draw_bboxes(image, bboxes)
    #cv2.imwrite("demo_out.jpg", image)
    return image, bboxes

def record(width, height):
    frame_rate = 10.0 # フレームレート
    size = (width, height) # 動画の画面サイズ
    
    fmt = cv2.VideoWriter_fourcc(*"XVID") # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter('./outtest.mp4', fmt, frame_rate, size) # ライター作成
    return writer

def main():
    args = sys.argv

    detector = CornerNet_Squeeze()
    cam(args[1], detector)
    #cam(args[2], detector)

if __name__ == "__main__":
    main()
