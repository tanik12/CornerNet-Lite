import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from core.detectors import CornerNet_Squeeze
from core.vis_utils import draw_bboxes, extract_specific_object
import sys
import glob
import os

from extract_color import extract_color_info
from  model import dir_check, load_model, inference
import numpy as np

def camera_open():
    try:
        for num in range(10):
            cap = cv2.VideoCapture(num)
            if cap.isOpened():
                print("カメラデバイスが見つかりました。番号は" + str(num) + "番です。")
                return cap
    except:
        print("カメラデバイスが見つかりませんでした。")
        sys.exit()

def load_color4train(data_path):
    imgs_path = glob.glob(data_path)
    return imgs_path 

def cam(arg, detector):
    count = 1
    save_flag = False
    print(arg)

    current_path = os.getcwd()
    model_dirpath = current_path + "/model"
    clf = load_model(model_dirpath)

    if arg == "video":
        #cap = cv2.VideoCapture('/home/gisen/Documents/rosbag/2019-07-09-15-25-21.avi')
        cap = cv2.VideoCapture('/home/gisen/Documents/rosbag/out_short.mp4')
        width = int(cap.get(3))
        height = int(cap.get(4))
        writer = record(width, height)
    elif arg == "camera":
        cap = camera_open()   
        cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # カメラ画像の横幅を1280に設定
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # カメラ画像の縦幅を720に設定
    elif arg == "make_color4train":
        cap = camera_open()  
        cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # カメラ画像の横幅を1280に設定
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # カメラ画像の縦幅を720に設定
        data_path = "/home/gisen/Documents/own_dataset/traffic_light_dataset/traffic_light/*"
        imgs_path = load_color4train(data_path)
        save_flag = True

    #while True:
    while (cap.isOpened()):
        if arg == "video" or arg == "camera":              
            # VideoCaptureから1フレーム読み込む
            ret, frame = cap.read()
            if not ret:
                print("画像の取得に失敗しました。")
                continue
              
            image, bboxes, bboxes_traffic, bboxes_pdstrn = obj_inference(detector, frame)
                               
            img_lists = []
            if bboxes_traffic.shape[0] > 0:
                try:
                    for bbox in bboxes_traffic:
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[2])
                        y2 = int(bbox[3])
    
                        trm_img = image[y1:y2,x1:x2]
                        img_lists.append([trm_img])
                except:
                    print("交通信号機のトリミングを試みましたが失敗しました")

            if bboxes_pdstrn.shape[0] > 0:
                try:
                    for bbox in bboxes_pdstrn:
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[2])
                        y2 = int(bbox[3])
                        
                        trm_img = image[y1:y2,x1:x2]
                        img_lists.append([trm_img])
                except:
                    print("歩行者信号機のトリミングを試みましたが失敗しました")

            if len(img_lists) > 0:
                res_data = extract_color_info(img_lists)
                #print("(r, g, b, h, s, v): ", res_data[0][4]) #Debug用

                for input_data in res_data:
                    input_data = np.array(input_data[4])
                    pred, label_name = inference(input_data,  clf)
                print("予想ラベル出力: ", label_name)
    
            if arg == "video":
                writer.write(image) # 画像を1フレーム分として書き込み
    
            # 加工なし画像を表示する
            cv2.imshow('Raw Frame', image)
            
            # キー入力でqを押したら終了する
            k = cv2.waitKey(1)
            if k == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
        else:
            for img_path in imgs_path:
                img_name = os.path.basename(img_path)
                img = cv2.imread(img_path)
                image, bboxes, _, _ = obj_inference(detector, img, count, image_name=img_name, flag=save_flag)
                count += 1
            break

    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

#Falseにするとトリミング画像を保存しない
def obj_inference(detector, image, count=1, image_name=None, flag=False):
    bboxes_traffic = ""
    bboxes_pdstrn = ""

    bboxes = detector(image)
    if flag:
        bboxes_traffic, bboxes_pdstrn = extract_specific_object(image, bboxes, count, image_name=image_name, flag=flag)
    else:
        bboxes_traffic, bboxes_pdstrn = extract_specific_object(image, bboxes, count, image_name=image_name, flag=flag)
    
        ###image  = draw_bboxes(image, bboxes)

    return image, bboxes, bboxes_traffic, bboxes_pdstrn

def record(width, height):
    frame_rate = 10.0 # フレームレート
    size = (width, height) # 動画の画面サイズ
    
    fmt = cv2.VideoWriter_fourcc(*"XVID") # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter('./outtest.avi', fmt, frame_rate, size) # ライター作成
    return writer

def main():
    args = sys.argv

    detector = CornerNet_Squeeze()
    cam(args[1], detector)
    #cam(args[2], detector)

if __name__ == "__main__":
    main()
