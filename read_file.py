import pandas as pd

df = pd.read_table('C:\\99999\\class4\\MNIST_FC\\mnist_image_label\\mnist_test_jpg_10000.txt', sep=' ')
df.to_csv('test_file.csv', index=False)