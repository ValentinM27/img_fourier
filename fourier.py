# Import des librairies
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as i, ImageOps

# Affichage CLI
from tqdm import tqdm

def main():
	# Traitement image
	image = i.open("cafe.jpg")
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

		plt.subplot(131)
		plt.title("Image originale convertie en gris")
		plt.imshow(pixels, cmap='gray', vmin=0, vmax=255)

		plt.subplot(132)
		plt.title("DFT 2D")
		plt.imshow(np.real(dft_2D), cmap='gray', vmin=0, vmax=255)

		plt.subplot(133)
		plt.title("IDFT 2D")
		plt.imshow(np.real(idft_2D), cmap='gray', vmin=0, vmax=255)

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

	# Si les valeurs de parties sont infiniment petites, on inverse les parties réelle et complexe
	for j in range(N) :
		if abs(dft_img[j].real) < (10**(-15)) :
			dft_img[j] = complex(0,dft_img[j].imag)
		if abs(dft_img[j].imag) < (10**(-15)) :
			dft_img[j] = complex(dft_img[j].real,0)

	# On retourne l'absoule
	return dft_img

'''
Transformée de fourier discrète 2D
'''
def discrete_fourier_transform_2D(image):
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
		idft_img[u] = somme.real


	# Si les valeurs de parties sont infiniment petites, on inverse les parties réelle et complexe
	for j in range(N) :
		if abs(idft_img[j].real) < (10**(-15)) :
			idft_img[j] = complex(0,idft_img[j].imag)
		if abs(idft_img[j].imag) < (10**(-15)) :
			idft_img[j] = complex(idft_img[j].real,0)

	# On retourne l'absoule
	return idft_img

'''
Inverse de la transformée de Fourier discrète 2D
'''
def invert_discrete_fourier_transform_2D(image):
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

if __name__=="__main__":
	main()
