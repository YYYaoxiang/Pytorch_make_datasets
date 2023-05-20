from PIL import Image
import numpy as np

# picture=np.zeros([600,800,3],np.uint8)
# picture[:,:,2]=np.ones([600,800])*1
# picture[18:21,100:300,0]=np.ones([3,200])*255
# picture[18:21,100:300,1]=np.ones([3,200])*255
# picture[18:21, 100:300, 2] = np.ones([3, 200]) * 255
# image_02=Image.fromarray(picture)
# # image_02.save('whatever.bmp')
# # image_02.show()
# print(image_02.mode)
# picture2=image_02.convert('L')
# print(picture2.mode)
# matrx=np.array(picture2)
# print(matrx)
#



matrix1=Image.open("C:\\99999\\class4\\MNIST_FC\\mnist_image_label\\mnist_test_jpg_10000\\0_7.jpg")
matrix2=np.asarray(matrix1)  #图片转张量
print(matrix2)

