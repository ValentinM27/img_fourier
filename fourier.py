# Import des librairies
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as i, ImageOps

# Affichage CLI
from tqdm import tqdm

def main():
	image = i.open("cafe.jpg")
	gray_image = ImageOps.grayscale(image)
	pixels = list(gray_image.getdata())
	width, height = image.size
	pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

	dft_2D = discrete_fourier_transform_2D(pixels);
	plt.title("DFT 2D")
	plt.imshow(dft_2D, cmap='gray', vmin=0, vmax=255)
	plt.show()

'''
Fonction de transposition
'''
def transposition(image):
    return list(map(list,zip(*image)))

'''
Transformée de fourier discrète 1D
'''
def discrete_fourier_transform_1D(image):
	N = len(image)
	dft_img = N*[0]

	omega = np.exp(-2j*np.pi/N)

	for u in range(N):
		somme = 0
		for x in range(N):
			somme += image[x]*omega**(x*u)

		# On associe les parties réelle et imaginaire
		dft_img[u] = round(somme.real, 4) + round(somme.imag, 4)

	# On retourne l'absoule
	return np.abs(dft_img)

'''
Transformée de fourier discrète 2D
'''
def discrete_fourier_transform_2D(image):
	m = len(image)
	n = len(image[0])

	dft_img = m*[n*[0]]

	print("Traitement des lignes")
	for i in tqdm(range(m)):
		dft_img[i] = discrete_fourier_transform_1D(image[i])

	dft_img = transposition(dft_img)

	print("Traitement des colonnes")
	for i in tqdm(range(n)):
		dft_img[i] = discrete_fourier_transform_1D(dft_img[i])

	return transposition(dft_img)

if __name__=="__main__":
	main()
