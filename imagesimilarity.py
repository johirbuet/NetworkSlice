from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	#m =0
	s = ssim(imageA, imageB)
	d = (1 - s)/2.0
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f, DSSIM:  %0.2f" % (m, s, d))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()

fig = plt.figure("Images")

original = cv2.imread("zero0.png")
proposed = cv2.imread("zero1.png")
random =  cv2.imread("zero0.png")
 
# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
proposed = cv2.cvtColor(proposed, cv2.COLOR_BGR2GRAY)
random = cv2.cvtColor(random, cv2.COLOR_BGR2GRAY)

fig = plt.figure("Images")
images = ("Original", original), ("proposed", proposed), ("Random", random)
 
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
 
# show the figure
plt.show()
 
# compare the images
compare_images(original, original, "Original vs. Original")
compare_images(original, proposed, "Original vs. Proposed")
compare_images(original, random, "Original vs. Random")
