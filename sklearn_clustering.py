from sklearn.cluster import SpectralClustering, KMeans
import numpy as np


X1 = np.array([0]*300 + [1]*300 + [2]*130).reshape(730, -1)
X2 = np.random.rand(730, 1)*10
# X1 = np.array([[ 10.02244444],
# [ 0. ],
# [ 10.02244444],
# [ 10.02244444],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 10.02244444],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 57.798 ],
# [ 90.37333333],
# [ 52.28692308],
# [ 75.10666667],
# [ 83.26833333],
# [ 46.767 ],
# [ 60.602 ],
# [ 53.65375 ],
# [ 43.734 ],
# [ 41.347 ],
# [ 83.74875 ],
# [ 37.598 ],
# [ 0. ],
# [ 41.1 ],
# [ 74.30076923],
# [ 62.748 ],
# [104.402 ],
# [ 42.501 ],
# [ 36.597 ],
# [ 10.02244444],
# [ 45.365 ],
# [ 10.02244444],
# [ 38.802 ],
# [ 35.3775 ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 10.02244444],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [146.05411765],
# [140.602 ],
# [ 61.198 ],
# [153.27 ],
# [127.898 ],
# [153.05444444],
# [ 69.798 ],
# [ 0. ],
# [128.634 ],
# [ 30.89 ],
# [ 10.02244444],
# [265.178125 ],
# [176.00071429],
# [297.69083333],
# [248.0575 ],
# [349.725 ],
# [326.396 ],
# [301.353 ],
# [204.101 ],
# [159.365 ],
# [261.808 ],
# [229.80076923],
# [192.69 ],
# [202.545 ],
# [276.933 ],
# [388.278 ],
# [355.034 ],
# [ 0. ],
# [400.638 ],
# [ 0. ],
# [293.438 ],
# [216.298 ],
# [230.783 ],
# [326.252 ],
# [124.824 ],
# [264.203 ],
# [199.758 ],
# [135.233 ],
# [318.098 ],
# [143.899 ],
# [128.155 ],
# [ 10.02244444],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 10.02244444],
# [131.6 ],
# [ 0. ],
# [ 30.70230769],
# [156.175 ],
# [ 42.31 ],
# [ 39.15 ],
# [ 24.40625 ],
# [ 92.966 ],
# [ 35.565 ],
# [ 22.999 ],
# [ 97.16230769],
# [ 49.323 ],
# [ 36.405 ],
# [ 68.5 ],
# [ 10.02244444],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 10.02244444],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 10.02244444],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 10.02244444],
# [ 10.02244444],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 45.952 ],
# [ 83.86083333],
# [ 27.01769231],
# [ 34.49833333],
# [ 25.4 ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 10.02244444],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 35.975 ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 10.02244444],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [119.949 ],
# [ 0. ],
# [ 0. ],
# [125.07 ],
# [132.398 ],
# [124.647 ],
# [124.6 ],
# [161.28285714],
# [ 0. ],
# [ 0. ],
# [154.45875 ],
# [118.6 ],
# [123.71125 ],
# [ 0. ],
# [125.74666667],
# [146.86230769],
# [141.12230769],
# [184.723 ],
# [244. ],
# [174.026 ],
# [138.585 ],
# [165.977 ],
# [ 0. ],
# [121.186 ],
# [ 0. ],
# [104.946 ],
# [ 0. ],
# [ 0. ],
# [143.624 ],
# [130.39416667],
# [140.868 ],
# [101.199 ],
# [ 0. ],
# [ 0. ],
# [ 98.935 ],[106.49 ],
# [136.417 ],
# [ 0. ],
# [212.583 ],
# [138.84375 ],
# [ 0. ],
# [109.7475 ],
# [ 63.341 ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 81.431 ],
# [ 58.258 ],
# [ 21.722 ],
# [ 10.02244444],
# [ 0. ],
# [ 10.02244444],
# [ 10.02244444],
# [ 10.02244444]])
# x2 = np.array([[ 39.599 ],
# [ 50.702 ],
# [ 17.94091026],
# [ 27. ],
# [ 30.3 ],
# [ 17.94091026],
# [ 17.94091026],
# [ 0. ],
# [ 0. ],
# [ 17.94091026],
# [ 17.94091026],
# [ 0. ],
# [ 0. ],
# [ 17.94091026],
# [ 17.94091026],
# [ 17.94091026],
# [ 0. ],
# [ 0. ],
# [ 25.6 ],
# [ 40.061 ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],[ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 46.653 ],
# [ 75.381 ],
# [ 27.149 ],
# [ 28.698 ],
# [ 0. ],
# [ 36.8 ],[ 32.399 ],
# [ 66.401 ],
# [ 67.502 ],
# [ 64.798 ],
# [ 33.601 ],
# [ 54.386 ],
# [ 51.328 ],
# [ 90.3 ],
# [ 91.28571429],
# [ 93.4525 ],
# [ 93.71142857],
# [ 73.404 ],
# [ 0. ],
# [ 63.9 ],
# [ 72. ],
# [ 98.22384615],
# [ 27.34615385],
# [ 84.996 ],
# [100.902 ],
# [ 79.18769231],
# [ 98.39875 ],
# [ 31.6 ],
# [ 53.84833333],
# [ 62.69284615],
# [ 60.155 ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 17.94091026],
# [ 17.94091026],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 39.4 ],
# [ 75.997 ],
# [ 44.303 ],
# [ 17.94091026],
# [ 17.94091026],
# [ 59.33666667],
# [ 54.504 ],
# [ 0. ],
# [ 70.24 ],
# [ 67.33416667],
# [ 46.198 ],
# [103.23538462],[ 89.381 ],
# [ 87.892 ],
# [105.881 ],
# [ 52.821 ],
# [ 47.468 ],
# [ 0. ],
# [ 54.035 ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 47.498 ],
# [ 27.618 ],
# [ 0. ],
# [ 39.13083333],
# [ 32.492 ],
# [ 28.288 ],
# [ 17.94091026],
# [ 41.95833333],
# [ 0. ],
# [ 0. ],
# [ 41.22923077],
# [ 0. ],
# [ 0. ],
# [ 63.91833333],
# [ 46.1 ],
# [104.842 ],
# [116.158 ],
# [122.437 ],
# [218.447 ],
# [192.173 ],
# [ 42.6 ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 0. ],
# [ 67.38583333],
# [ 93.27076923],
# [ 49.5 ],
# [ 0. ],
# [ 67.65538462]])
X3 = np.array([[ 28.087 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 11.338 ],
[ 10.958 ],
[ 23.21 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 10.01142857],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 11.05769231],
[ 0. ],
[ 6.32299292],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 13.37416667],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 11.6125 ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 10.92222222],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 12.88 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 14.125 ],
[ 15.918 ],
[ 13.58166667],
[ 0. ],
[ 0. ],
[ 12.94833333],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 11.465 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 11.31 ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 14.45714286],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 11.97444444],
[ 0. ],
[ 0. ],
[ 17.949 ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 12.195 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 11.379 ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 16.489 ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 11.71285714],
[ 6.32299292],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 12. ],
[ 0. ],
[ 0. ],
[ 20.854 ],
[ 0. ],
[ 0. ],
[ 13.214 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 12.95833333],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 42.91 ],
[104.236 ],
[115.55666667],
[ 0. ],
[171.6675 ],
[ 0. ],
[205.9015 ],
[ 0. ],
[ 0. ],
[ 0. ],
[241.385 ],
[196.092 ],
[ 0. ],
[162.993 ],
[243.558 ],
[152.168 ],
[165.21875 ],
[ 88.98375 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 29.87285714],
[ 0. ],
[226.11333333],
[ 0. ],
[312.09 ],
[270.921 ],
[282.289 ],
[ 0. ],
[240.625 ],
[ 0. ],
[160.343 ],
[136.526 ],
[ 42.54192857],
[281.923 ],
[246.981 ],
[ 0. ],
[ 55.211 ],
[ 0. ],
[ 10.059 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[148.8375 ],
[196.677 ],
[214.174 ],
[ 0. ],
[ 0. ],
[ 64.47142857],[ 86.87166667],
[ 0. ],
[ 0. ],
[ 0. ],
[ 53.006 ],
[ 43.72375 ],
[ 34.221 ],
[ 54.28428571],
[ 52.534 ],
[ 57.619 ],
[ 18.2675 ],
[ 31.72 ],
[ 13.98333333],
[ 0. ],
[ 54.534 ],
[ 0. ],
[ 19.209 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 20.67 ],
[ 11.83933333],
[ 12.699 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 20.694 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 14.4875 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.01866667],
[ 0. ],
[ 0. ],
[ 14.43333333],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 14.08571429],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 10.87 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 10.408 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 12.00714286],[ 0. ],
[ 0. ],
[ 9.675 ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 6.32299292],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 12.956 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 11.439 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 10.799 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 43.61666667],
[ 0. ],
[ 0. ],
[ 67.07666667],
[ 68.63833333],
[ 34.785 ],
[169.145 ],
[ 64.88 ],
[215.887 ],
[208.115 ],
[ 68.68 ],
[ 0. ],
[ 0. ],
[ 0. ],
[101.64625 ],
[ 0. ],
[ 0. ],
[216.88333333],
[183.02285714],
[158.032 ],
[209.226 ],
[181.583 ],
[ 0. ],
[ 0. ],
[259.47833333],
[445.46692857],
[315.904 ],
[ 0. ],
[707.911 ],
[491.3405 ],
[393.348 ],
[531.05285714],
[553.98571429],
[490.376 ],
[428.205 ],
[120.47428571],
[ 0. ],
[105.443 ],
[102.025 ],
[116.292 ],
[128.781 ],
[ 90.097 ],
[162.79 ],
[ 91.61388889],
[141.74125 ],
[ 27.41375 ],
[ 0. ],
[ 0. ],
[ 34.33166667],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 13.76 ],
[ 0. ],[ 0. ],
[206.681 ],
[133.394 ],
[ 91.924 ],
[198.757 ],
[164.715 ],
[156.114 ],
[ 0. ],
[ 0. ],
[167.927 ],
[178.035 ],
[153.069 ],
[197.913 ],
[185.908 ],
[196.648 ],
[229.603 ]])
X4 = np.array([[ 57.42 ],
[ 0. ],
[ 95.535 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 146.301 ],
[ 82.598 ],
[ 66.899 ],
[ 101.28 ],
[ 80.399 ],
[ 147.148 ],
[ 131.499 ],
[ 247.071 ],
[ 140.1 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 332.85 ],
[ 346.099 ],
[ 355.55333333],
[ 368.928 ],
[ 321.287 ],
[ 271.169 ],
[ 311.75384615],
[ 371.483 ],
[ 265.118 ],
[ 165.642 ],
[ 207.719 ],
[ 274.39 ],
[ 218.208 ],
[ 17.52545402],
[ 819.38428571],
[1854.7295 ],
[1200.50555556],[ 767.94833333],
[2717.44 ],
[2826.65714286],
[2718.693 ],
[2612.483 ],
[1530.44166667],
[2643.43214286],
[3286.15666667],
[2688.6025 ],
[3036.629 ],
[2557.798 ],
[3102.446 ],
[3478.08 ],
[2950.181 ],
[2124.783 ],
[1744.527 ],
[1535.126 ],
[1237.491 ],
[2134.128 ],
[1976.838 ],
[2150.157 ],
[2036.933 ],
[1415.328 ],
[1151.506 ],
[1575.319 ],
[ 572.417 ],
[ 644.099 ],
[ 365.461 ],
[ 185.63375 ],
[ 149.98375 ],
[ 62.081 ],
[ 375.581 ],
[ 17.52545402],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 52.065 ],
[ 98.872 ],
[ 77.749 ],
[ 125.62 ],
[ 731.686 ],
[1083.639 ],
[ 803.331 ],
[ 900.708 ],
[ 903.061 ],
[ 525.628 ],
[ 442.23538462],
[ 427.85142857],
[ 438.96882353],
[ 729.624 ],
[ 423.732 ],
[ 519.257 ],
[ 761.003 ],
[ 420.601 ],
[ 376.029 ],
[ 358.432 ],
[ 260.672 ],
[ 270.624 ],
[ 103.724 ],
[ 17.52545402],
[ 50.266 ],
[ 17.52545402],
[ 67.296 ],
[ 79.843 ],
[ 146.131 ],
[ 435.86 ],
[ 625.42166667],
[ 324.198 ],
[ 488.96428571],
[ 607.027 ],
[ 720.471 ],
[ 604.33357143],
[ 715.78583333],
[1038.985 ],
[ 894.07428571],[1262.80285714],
[1228.078 ],
[1432.548 ],
[1230.48 ],
[1067.185 ],
[ 923.193 ],
[1062.05428571],
[1239.919 ],
[1100.481 ],
[ 779.91 ],
[ 818.71166667],
[ 740.195 ],
[ 674.181 ],
[ 844.3125 ],
[ 629.117 ],
[ 648.776 ],
[ 756.336 ],
[1036.171 ],
[ 894.845 ],
[1141.434 ],
[ 801.005 ],
[ 930.882 ],
[ 696.113 ],
[ 965.227 ],
[ 922.04538462],
[ 868.66923077],
[ 591.508 ],
[ 617.714 ],
[ 190.775 ],
[ 152.455 ],
[ 195.753 ],
[ 154.74833333],
[ 156.27083333],
[ 206.639 ],
[ 139.95 ],
[ 184.445 ],
[ 165.825 ],
[ 193.574 ],
[ 121.8925 ],
[ 120.9725 ],
[ 97.394 ],
[ 106.74125 ],
[ 121.7 ],
[ 86.986 ],
[ 120.623 ],
[ 91.75833333],
[ 116.21833333],
[ 205.94625 ],
[ 145.542 ],
[ 70.026 ],
[ 85.238 ],
[ 58.179 ],
[ 17.52545402],
[ 197.012 ],
[ 17.52545402],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 17.52545402],[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 17.52545402],
[ 0. ],
[ 17.52545402]])
X5 = np.array([[ 5.12981775],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 5.12981775],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 5.12981775],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 5.12981775],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 75.21416667],
[ 72.57 ],
[ 76.594 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 29.813 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 25.39 ],
[ 33.388 ],
[ 29.176 ],
[ 0. ],
[ 0. ],
[ 38.25 ],[ 0. ],
[ 77.548 ],
[ 69.51285714],
[ 0. ],
[ 0. ],
[ 49.665 ],
[ 62.723 ],
[ 71.20066667],
[ 67.46647059],
[ 0. ],
[ 0. ],
[ 40.87375 ],
[ 63.82714286],
[ 0. ],
[ 40.325 ],
[ 18.51916667],
[ 0. ],
[ 38.078 ],
[ 26.31125 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 5.12981775],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 5.12981775],
[ 0. ],
[ 0. ],
[ 0. ],
[ 5.12981775],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 12.95235294],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 24.08166667],
[ 0. ],
[ 24.599 ],
[ 23.37625 ],
[ 0. ],
[ 11.787 ],
[ 32.674 ],
[ 24.339 ],[ 17.843 ],
[ 14.258 ],
[ 19.863 ],
[ 20.233 ],
[ 16.829 ],
[ 17.264 ],
[ 15.959 ],
[ 20.241 ],
[ 16.024 ],
[ 14.83166667],
[ 20.96714286],
[ 19.36 ],
[ 56.812 ],
[ 37.688 ],
[ 46.404 ],
[ 15.996 ],
[ 29.732 ],
[ 96.0845 ],
[137.188 ],
[ 20.92 ],
[149.897 ],
[ 32.8055 ],
[ 56.7185 ],
[197.764 ],
[ 75.749 ],
[ 43.35785714],
[ 21.977 ],
[ 63.717 ],
[ 94.31076923],
[ 95.81461538],
[ 62.27333333],
[ 21.064 ],
[ 12.699 ],
[ 0. ],
[ 18.56125 ],
[ 5.12981775],
[ 0. ],
[ 30.32 ],
[ 24.87 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 5.12981775],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 5.12981775],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 12.099 ],
[ 0. ],
[ 5.12981775],
[ 0. ],
[ 0. ],[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 11.7 ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 0. ],
[ 5.12981775],
[ 0. ],
[ 0. ],
[ 5.12981775]])
X = np.array([X1, X2, X3, X4, X5])


b_Spectral = []
for i in range(X.shape[0]):
    try:
        clustering = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0, n_jobs=-1,
                                        affinity='poly', n_neighbors=int(len(X[i])/3/6), degree=3, coef0=1).fit(X[i])
        b_Spectral.append(clustering.labels_)
    except:
        print('聚类函数出错', '\n')
print(b_Spectral, np.shape(b_Spectral), '\n')  # labels序号的大小与labels中数值的大小无关


b_kmeans = []
for i in range(X.shape[0]):
    try:
        clustering = KMeans(n_clusters=3, n_init=100, random_state=0).fit(X[i])
        b_kmeans.append(clustering.labels_)
    except:
        print('聚类函数出错', '\n')
print(b_kmeans, np.shape(b_kmeans), '\n')  # labels序号的大小与labels中数值的大小无关


for i in range(len(b_Spectral)):
    print(sum(b_Spectral[i] == b_kmeans[i]) == len(b_Spectral[i]))