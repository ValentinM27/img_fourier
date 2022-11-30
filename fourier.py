# Import des librairies
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image as i, ImageOps

# Affichage CLI
from tqdm import tqdm
from sys import exit

def main():
	# Traitement image
	image = i.open("lena.jpg")
	gray_image = ImageOps.grayscale(image)
	pixels = list(gray_image.getdata())
	width, height = image.size
	pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

	choice = 0
	choice = int(input("Choix du programme : "))

	if choice == 1:
		# dft 2D
		dft_2D = discrete_fourier_transform_2D(pixels);

		idft_2D = invert_discrete_fourier_transform_2D(dft_2D)
		arr_idft_2D = np.array(idft_2D)
		image_idft_2D = i.fromarray(arr_idft_2D).rotate(180)

		plt.subplot(131)
		plt.title("Image originale convertie en gris")
		plt.imshow(pixels, cmap='gray', vmin=0, vmax=255)

		plt.subplot(132)
		plt.title("DFT 2D")
		plt.imshow(np.abs(dft_2D), cmap='gray', vmin=0, vmax=255)

		plt.subplot(133)
		plt.title("IDFT 2D")
		plt.imshow(image_idft_2D, cmap='gray', vmin=0, vmax=255)

		plt.show()

	if choice == 2:
		# fft 2D
		fft_2D = fast_fourier_transform_2D(pixels);
		ifft_2D = invert_fast_fourier_transform_2D(fft_2D)

		plt.subplot(131)
		plt.title("Image originale convertie en gris")
		plt.imshow(pixels, cmap='gray', vmin=0, vmax=255)

		plt.subplot(132)
		plt.title("FFT 2D")
		plt.imshow(np.abs(fft_2D), cmap='gray', vmin=0, vmax=255)

		plt.subplot(133)
		plt.title("IFFT 2D")
		plt.imshow(ifft_2D, cmap='gray', vmin=0, vmax=255)

		plt.show()

'''
Fonction de transposition
'''
def transposition(image):
    return list(map(list,zip(*image)))

'''
Fonction de normalisation 2D
'''
def normalise2d(n):
	size_x, size_y = len(n), len(n[0])
	for i in range(size_x):
		for j in range(size_y):
			n[i][j] = round(n[i][j].real/(size_x*size_y),4)
	return n

'''
Transformée discrètes
'''

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
	return dft_img

'''
Transformée de fourier discrète 2D
'''
def discrete_fourier_transform_2D(image):
	print("---- DFT 2D ----")
	m = len(image)
	n = len(image[0])

	dft_img = m*[n*[0]]

	print("Traitement des colonnes")
	for i in tqdm(range(m)):
		dft_img[i] = discrete_fourier_transform_1D(image[i])

	dft_img = transposition(dft_img)

	print("Traitement des lignes")
	for i in tqdm(range(n)):
		dft_img[i] = discrete_fourier_transform_1D(dft_img[i])

	return transposition(dft_img)

'''
inverses discrètes
'''

'''
Inverse de la transformée de Fourier discrète 1D
'''
def invert_discrete_fourier_transform_1D(image):
	N = len(image)
	idft_img = N*[0]

	omega = np.exp(2j*np.pi/N)

	for u in range(N):
		somme = 0
		for x in range(N):
			somme += image[x]*omega**(x*u)

		# On associe les parties réelle et imaginaire
		idft_img[u] = round(somme.real, 4) + round(somme.imag, 4)

	# On retourne l'absoule
	return idft_img

'''
Inverse de la transformée de Fourier discrète 2D
'''
def invert_discrete_fourier_transform_2D(image):
	print("---- IDFT 2D ----")
	m = len(image)
	n = len(image[0])

	idft_img = m*[n*[0]]

	print("Traitement des colonnes")
	for i in tqdm(range(m)):
		idft_img[i] = invert_discrete_fourier_transform_1D(image[i])

	idft_img = transposition(idft_img)

	print("Traitement des lignes")
	for i in tqdm(range(n)):
		idft_img[i] = invert_discrete_fourier_transform_1D(idft_img[i])

	return normalise2d(transposition(idft_img))

'''
Transformée rapide
'''

'''
Trasnformée de fourier rapide 1D
'''
def fast_fourier_transform_1D(image):
	# Taille de l'image
	N = len(image)
	# Image finale
	fft_image_1D = N*[0]


	# Si l'image ne fait que un pixel, sa fft est correpondant à ce pixel
	if N == 1:
		return image

	# Si le nombre de pixel n'est pas une puissance de 2 on arrête le traitement
	if not math.log2(N).is_integer() :
		print("L'image comporte",N,"pixels qui n'est pas une puissance de 2")
		exit()

	# Pour des raisons de performance on n'effectu le calcul qu'une seule fois
	omega = np.exp(-2j*np.pi/N)

	# On coupe l'image en deux groupe, pair et impair
	image_pair = image[::2]
	image_impair = image[1::2]

	# résultat
	fft_image_pair = fast_fourier_transform_1D(image_pair)
	fft_image_impair = fast_fourier_transform_1D(image_impair)

	# Reconstruction de l'image finale
	for j in range(N//2):
		fft_image_1D[j] = fft_image_pair[j] + (omega**j)*fft_image_impair[j]
		fft_image_1D[j+N//2] = fft_image_pair[j] - (omega**j)*fft_image_impair[j]

	return fft_image_1D

def fast_fourier_transform_2D(image):
	print("---- FFT 2D ----")
	m = len(image)
	n = len(image[0])

	fft_img = m*[n*[0]]

	print("Traitement des colonnes")
	for i in tqdm(range(m)):
		fft_img[i] = fast_fourier_transform_1D(image[i])

	fft_img = transposition(fft_img)

	print("Traitement des lignes")
	for i in tqdm(range(n)):
		fft_img[i] = fast_fourier_transform_1D(fft_img[i])

	return transposition(fft_img)

'''
Transformée de Fourier rapide inverse
'''

'''
Transformée de Fourier rapide 1D inverse
'''
def invert_fast_fourier_transform_1D(image):
	# Taille de l'image
	N = len(image)
	# Image finale
	ifft_image_1D = N*[0]

	# Si l'image ne fait que un pixel, sa ifft est correpondant à ce pixel
	if N == 1:
		return image

	# Si le nombre de pixel n'est pas une puissance de 2 on arrête le traitement
	if not math.log2(N).is_integer() :
		print("L'image comporte",N,"pixels qui n'est pas une puissance de 2")
		exit()

	# Pour des raisons de performance on n'effectu le calcul qu'une seule fois
	omega = np.exp(2j*np.pi/N)

	# On coupe l'image en deux groupe, pair et impair
	image_pair = image[::2]
	image_impair = image[1::2]

	# résultat
	ifft_image_pair = invert_fast_fourier_transform_1D(image_pair)
	ifft_image_impair = invert_fast_fourier_transform_1D(image_impair)

	# Reconstruction de l'image finale
	for j in range(N//2):
		ifft_image_1D[j] = ifft_image_pair[j] + (omega**j)*ifft_image_impair[j]
		ifft_image_1D[j+N//2] = ifft_image_pair[j] - (omega**j)*ifft_image_impair[j]

	return ifft_image_1D

'''
Transformée de Fourier rapide 2D inverse
'''
def invert_fast_fourier_transform_2D(image):
	print("---- IFFT 2D ----")
	m = len(image)
	n = len(image[0])

	ifft_image = m*[n*[0]]

	print("Traitement de colonnes")
	for i in tqdm(range(m)):
		ifft_image[i] = invert_fast_fourier_transform_1D(image[i])

	ifft_image = transposition(ifft_image)

	print("Traitement des lignes")
	for i in tqdm(range(n)):
		ifft_image[i] = invert_fast_fourier_transform_1D(ifft_image[i])

	return normalise2d(transposition(ifft_image))

if __name__=="__main__":
	main()
