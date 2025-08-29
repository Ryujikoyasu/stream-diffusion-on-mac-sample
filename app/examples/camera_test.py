import cv2
import numpy as np


def main():
    # カメラデバイスの選択と接続確認
    for camera_id in range(10):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"カメラID {camera_id} が利用可能です")
                break
            cap.release()
    else:
        print("利用可能なカメラが見つかりませんでした")
        return

    window_name = "Camera Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("カメラからの読み取りに失敗しました")
                continue

            # フレームを表示
            cv2.imshow(window_name, frame)

            # qキーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("プログラムを終了します")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 