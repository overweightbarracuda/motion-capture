import cv2, os
# from cvzone.PoseModule import PoseDetector
imgs = []
# for i in range(1,35):
#     vid_name = "jab"+str(i)

#     cap = cv2.VideoCapture("./jab/"+vid_name+".mp4")

#     detector = PoseDetector()
#     posList = []
#     while True:
#         try:
#             success, img = cap.read()
#             img = cv2.resize(img, (800,400))
#             img = detector.findPose(img)
#             lmList, bboxInfo = detector.findPosition(img)

#             if(bboxInfo):
#                 lmString = ""
#                 for lm in lmList:
#                     lmString += f"{lm[1]},{img.shape[0]-lm[2]},{lm[3]};"
#                 posList.append(lmString[:-1])
#             print(len(posList))

#             cv2.imshow("Image",img)
#             key = cv2.waitKey(1)
#             if key == ord("s"):
#                 break


#         except:
#             print("no img left, ok please thank you very much love")
#             with open("./jabt/"+vid_name, "w") as f:
#                     f.writelines(["%s\n" % pos for pos in posList])
            
#             break

# for f in frames:
#     for i in range(0,33):
#         bpy.data.collections["points"].objects["s"+str(i)].location = (locs[f][i][0], locs[f][i][1], locs[f][i][2])
#         bpy.data.collections["points"].objects["s"+str(i)].keyframe_insert(data_path="location", frame=f)



for vid in os.listdir("./jabt/"):
    loc = []
    with open("./jabt/"+vid) as f:
        for line in f.readlines():
            loc.append([i.split(",") for i in line.split(";")])
    imgs.append(loc)
print(imgs[0][0][0])

# loc = []
# with open("jab1") as f:
#     for line in f.readlines():
#         loc.append([i.split(",") for i in line.split(";")])
