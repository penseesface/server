import onnxruntime as rt
import numpy as np
import cv2

sess = rt.InferenceSession('./model.onnx')


input_name = sess.get_inputs()[0].name
print("input name1:", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape1:", input_shape)



input_type = sess.get_inputs()[0].type
print("input type:", input_type)


output_name = sess.get_outputs()[0].name
print("output name:", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape:", output_shape)
output_type = sess.get_outputs()[0].type
print("output type:", output_type)

output_name = sess.get_outputs()[1].name
print("output name:", output_name)
output_shape = sess.get_outputs()[1].shape
print("output shape:", output_shape)
output_type = sess.get_outputs()[1].type
print("output type:", output_type)



output_name = sess.get_outputs()[2].name
print("output name:", output_name)
output_shape = sess.get_outputs()[2].shape
print("output shape:", output_shape)
output_type = sess.get_outputs()[2].type
print("output type:", output_type)




# runing output

# input name1:    actual_input_1
# input shape1:   [1, 3, 128, 128]   
# img format:     BGR  preprocess:  imgs = imgs / 127.5 - 1.

# input name2:    actual_input_2
# input shape2:   [1, 1, 1, 1]
# input type:     tensor(float)
# if ID photo:   value is 1  else  capture photo value is 0

# output name:    output1  
# output shape:   [1, 512]
# output type:    tensor(float)



# #test on images

# img1 = cv2.imread('01.jpg')
# img1 = cv2.resize(img1,(128,128))
# img2 = cv2.imread('02.jpg')
# img2 = cv2.resize(img2,(128,128))
# img3 = cv2.imread('03.jpg')
# img3 = cv2.resize(img3,(128,128))



# #inference on ID phtots or gallery photos
# img1 = np.float32([img1])
# img1 = np.transpose(img1, [0,3,1,2])
# ones = np.ones((1, 1, 1, 1)).astype(np.float32)
# img1 = img1 / 127.5 - 1.
# feats = sess.run([output_name], {'actual_input_1': img1,'actual_input_2': ones})
# feats = np.array(feats[0])
# feat1 = feats / np.linalg.norm(feats, axis=1, keepdims=True)

# print(feat1[0].shape)




# #inference on capture photos or probe photos
# img2 = np.float32([img2])
# img2 = np.transpose(img2, [0,3,1,2])
# ones = np.zeros((1, 1, 1, 1)).astype(np.float32) #all zeros
# img2 = img2 / 127.5 - 1.
# feats = sess.run([output_name], {'actual_input_1': img2,'actual_input_2': ones})
# feats = np.array(feats[0])
# feat2 = feats / np.linalg.norm(feats, axis=1, keepdims=True)

# print(feat2[0].shape)



# img3 = np.float32([img3])
# img3 = np.transpose(img3, [0,3,1,2])
# ones = np.zeros((1, 1, 1, 1)).astype(np.float32)  #all zeros
# img3 = img3 / 127.5 - 1.
# feats = sess.run([output_name], {'actual_input_1': img3,'actual_input_2': ones})
# feats = np.array(feats[0])
# feat3 = feats / np.linalg.norm(feats, axis=1, keepdims=True)

# print(feat3[0].shape)


# print('ONNX Inference scr:')
# print(np.sum(feat1[0]*feat2[0]))

# print(np.sum(feat1[0]*feat3[0]))
