import yolov5
# git clone https://huggingface.co/fcakyon/yolov5s-v7.0

model = yolov5.load('fcakyon/yolov5s-v7.0')

model.conf = 0.35
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 1000

img = r'C:\Users\Personal\Documents\projects\KICS Second Project\yolov5_image_annotate\cats.jpg'

result = model(img)
result = model(img, size=640)
result = model(img, augment= True)

# parse results
predictions = result.pred[0]
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

result.show()
result.save(save_dir = 'outcomes/')