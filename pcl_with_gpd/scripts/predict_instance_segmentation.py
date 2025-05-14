import os
from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils import ASSETS #官網範例程式引入模組
from ultralytics.utils.checks import check_yaml #官網範例程式引入模組
# import torch

class_names_dict={0: 'aluextru', 1: 'bin', 2: 'twpipe'}
print(class_names_dict)

class_colors = {
    0: (255, 0, 0),   # 類別 0 - 紅色
    1: (0, 255, 0),   # 類別 1 - 綠色
    2: (0, 0, 255),   # 類別 2 - 藍色
    # 添加更多類別顏色
}

image_path_test='/home/chen/圖片/detect.png'
# 設定已完成訓練的路徑
weight_path = r"Segmentation_Train/results/training_results5/weights/best.pt"
# model_path = trained_model_path + 'yolo11s-seg.pt'

# 加載 YOLO 模型
model = YOLO(weight_path)

# 設定測試圖片的路徑
test_image_path = r"/home/chen/Segmentation_Train/train_data/test/images"

# 確保測試圖片存在
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"測試圖片不存在: {test_image_path}")


img_list=[]
# 遍歷資料夾中的所有圖片

for image_name in os.listdir(test_image_path):
    # 確保是圖片檔案
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(test_image_path, image_name)
        img_list.append(image_path)
        # print(f"正在處理圖片: {image_path}")

def get_mask_data(image_path):
    results = model.predict(
        source=image_path,
        verbose=False,
        save=False,
        project="/home/chen/Segmentation_Train",
        name="view_1",
        show=False
    )

    # 讀取原始圖片
    original_image = cv2.imread(image_path)
    orig_h, orig_w = original_image.shape[:2]  # 原始尺寸
    mask_pixel_list = []
    class_id_list = []
    confidence_score_list = []

    for result in results:
        masks = result.masks
        classes = result.boxes.cls.cpu().numpy()
        confidence = result.boxes.conf.cpu().numpy()

        if masks is not None:
            for mask, cls, conf in zip(masks.data, classes, confidence):
                # mask: shape (640, 640) → 先轉為 uint8 binary
                mask = mask.cpu().numpy().astype(np.uint8) * 255

                # 👉 上採樣遮罩到原始解析度（如 1280×720）
                mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                confidence_score_list.append(conf)
                class_name = class_names_dict.get(int(cls), "Unknown")
                class_id_list.append(class_name)
                color = class_colors.get(int(cls), (255, 255, 255))

                # 擷取遮罩像素位置
                y_coords, x_coords = np.where(mask_resized > 0)
                pixel_coords = np.stack([x_coords, y_coords], axis=1)  # shape: (N, 2)
                mask_pixel_list.append(pixel_coords)

                # 👉 以下繪圖部分套用 resized 遮罩
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(original_image, contours, -1, color, thickness=1)

                overlay = original_image.copy()
                overlay[mask_resized > 0] = color
                alpha = 0.5
                original_image = cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0)

                # 可選：加上類別文字
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(original_image, class_name, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, color, 2, cv2.LINE_AA)

    return original_image, mask_pixel_list, class_id_list, confidence_score_list

def get_mask_data_accurate(image_path):
    results = model.predict(
        source=image_path,
        verbose=False,
        save=False,
        retina_masks=True,  # ← ✅ 確保輸出高解析度遮罩
        device='cpu'
    )

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    all_pixel_coords = []
    class_id_list = []
    confidence_score_list = []

    for result in results:
        masks = result.masks.data  # [n, h, w] torch.Tensor
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for i, (cls, conf) in enumerate(zip(classes, scores)):
            class_name = class_names_dict.get(int(cls), "Unknown")
            class_id_list.append(class_name)
            confidence_score_list.append(conf)

            # 提取單一 mask
            mask_tensor = masks[i]  # shape: (h, w)
            mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # 獲取遮罩像素點位置
            y, x = np.where(mask_resized > 0)
            pixel_coords = np.stack([x, y], axis=1)
            all_pixel_coords.append(pixel_coords)

            # 遮罩上色與疊加
            color = class_colors.get(int(cls), (255, 255, 255))
            overlay = image.copy()
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
            image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

            # 邊框與文字
            cv2.drawContours(image, contours, -1, color, thickness=2)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(image, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image, all_pixel_coords, class_id_list, confidence_score_list        

if __name__=="__main__":
    for _ in range(1):
        image,mask_pixel_list,class_id_list,confidence_score_list=get_mask_data_accurate(image_path_test)
        zip_list=list(zip(mask_pixel_list,class_id_list,confidence_score_list))
        print(zip_list[0])               

        # 顯示結果
        cv2.namedWindow("segmentation_result",cv2.WINDOW_NORMAL)
        cv2.resizeWindow('segmentation_result',640,480) #(寬,高)
        cv2.imshow("segmentation_result", image)
        key=cv2.waitKey(0)  # 按任意鍵繼續
        if key==ord('q'):
            cv2.destroyAllWindows()
            break

    

