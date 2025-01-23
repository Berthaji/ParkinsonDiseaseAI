import os
import random
from PIL import Image

def rotate_and_save_image(image, save_path, angle):
    """
    Ruota e salva l'immagine.
    
    :param image: L'immagine da ruotare.
    :param save_path: Percorso dove salvare l'immagine ruotata.
    :param angle: Angolo di rotazione.
    """
    rotated_image = image.rotate(angle)
    rotated_image.save(save_path)

def flip_and_save_image(image, save_path):
    """
    Applica un flip casuale (orizzontale o verticale) all'immagine e la salva.
    
    :param image: L'immagine da flippar.
    :param save_path: Percorso dove salvare l'immagine flip.
    """
    # Flip casuale orizzontale o verticale
    flip_type = random.choice(['horizontal', 'vertical'])
    if flip_type == 'horizontal':
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    
    flipped_image.save(save_path)

def preprocess_images(input_dir, rotation_angles=[90, 180, 270], num_rotations=1, apply_flip=True):
    """
    Ruota e applica il flip casuale alle immagini, aggiungendole al dataset originale.

    :param input_dir: Directory contenente il dataset originale (cartella 'spirali' e 'wave')
    :param rotation_angles: Lista di angoli per la rotazione delle immagini (es. [90, 180, 270])
    :param num_rotations: Numero di immagini ruotate da aggiungere per ogni immagine originale
    :param apply_flip: Se True, applica un flip casuale orizzontale o verticale
    """
    # Per ogni sottocartella (ad esempio 'sano' o 'malato')
    for class_folder in os.listdir(input_dir):
        class_folder_path = os.path.join(input_dir, class_folder)
        
        # Se è una cartella, prosegui
        if os.path.isdir(class_folder_path):
            # Scorri tutte le immagini nella cartella della classe
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)
                
                # Ignora i file che non sono immagini
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    continue

                # Se è un file immagine, aprilo
                if os.path.isfile(image_path):
                    try:
                        image = Image.open(image_path)
                    except Exception as e:
                        print(f"Impossibile aprire l'immagine {image_path}: {e}")
                        continue
                    
                    # Aggiungiamo il numero di immagini ruotate scelto per ogni immagine
                    for _ in range(num_rotations):
                        # Seleziona un angolo di rotazione casuale dalla lista
                        angle = random.choice(rotation_angles)
                        
                        # Genera un nuovo nome per l'immagine ruotata
                        rotated_image_name = f"{os.path.splitext(image_name)[0]}_rot_{angle}.png"
                        rotated_image_path = os.path.join(class_folder_path, rotated_image_name)
                        
                        # Salva l'immagine ruotata nella stessa cartella dell'immagine originale
                        rotate_and_save_image(image, rotated_image_path, angle)
                        
                        print(f"Immagine ruotata salvata: {rotated_image_path}")
                    
                    # Se il flip è abilitato, applica il flip casuale
                    if apply_flip:
                        flipped_image_name = f"{os.path.splitext(image_name)[0]}_flip.png"
                        flipped_image_path = os.path.join(class_folder_path, flipped_image_name)
                        
                        # Salva l'immagine flip
                        flip_and_save_image(image, flipped_image_path)
                        
                        print(f"Immagine flip salvata: {flipped_image_path}")
