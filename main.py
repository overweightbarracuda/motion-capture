import cv2
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture('PXL_20221119_195052395.TS.mp4')

detector = PoseDetector()
posList = []
while True:
    try:
        success, img = cap.read()
        img = cv2.resize(img, (800,400))
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img)

        if(bboxInfo):
            lmString = ""
            for lm in lmList:
                lmString += f"{lm[1]},{img.shape[0]-lm[2]},{lm[3]};"
            posList.append(lmString[:-1])
        print(len(posList))

        cv2.imshow("Image",img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            break


    except:
        print("no img left, ok please thank you very much love")
        with open("AnimFile2.txt", "w") as f:
                f.writelines(["%s\n" % pos for pos in posList])
        break

# for f in frames:
#     for i in range(0,33):
#         bpy.data.collections["points"].objects["s"+str(i)].location = (locs[f][i][0], locs[f][i][1], locs[f][i][2])
#         bpy.data.collections["points"].objects["s"+str(i)].keyframe_insert(data_path="location", frame=f)
