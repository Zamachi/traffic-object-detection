from PIL import ImageDraw, ImageFont
from typing import List
classes = ['bicycle', 'bus', 'car', 'motorbike', 'person']

def draw_image(img,boxes_data: List[str], show_image: bool=True, save_image: bool=False, denormalize:bool=True):
    """
    Iscrtava sliku `img` sa podacima o klasama i ogranicavajucim pravougaonicima iz `boxes_data`

    Parameters
    ----------
    img : Pillow.Image
        Slika ucitana uz pomoc Pillow biblioteke
    boxes_data : str[]
        Niz stringova koji sadrzi broj klase i podatke o ogranicavajucem pravougaoniku. Svaki string treba da bude oblika "`broj_klase x_centar y_centar box_width box_height`"
    show_image : bool=True
        Da li da prikaze sliku kada je iscrta. Podrazumevano je da.
    save_image : bool=True
        Da li da sacuva modifikovanu sliku, podrazumevano je ne.
    denormalize: bool=False
        Da li treba denormalizovati koordinate
    """

    draw = ImageDraw.Draw(img)

    for line in boxes_data:
        class_label, x_center, y_center, box_width, box_height = map(float, line.split())

        # Convert relative coordinates and sizes to absolute values
        if denormalize:
            width, height = img.size
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

        # Calculate the coordinates of the bounding box corners
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw the bounding box on the image
        outline_color = (0, 255, 0)  # Green color (RGB format)
        outline_thickness = 2
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=outline_thickness)

        # Write the class name above the rectangle
        text_color = (255, 0, 0)  # Red color (RGB format)
        font = ImageFont.load_default()  # You can customize the font if needed
        
        class_name = "Class: " + classes[int(class_label)]  # Convert class_label to a string if it's not already
        text_position = (x1, y1 - 15)  # Adjust the position as needed
        draw.text(text_position, class_name, fill=text_color, font=font)

    if show_image:
        img.show()
    elif save_image:
        img.save('output_img.jpg')
    else:
        return img

def read_txt_label_file(filepath: str):
    """
    Treba da ucita `labels` fajl sa `filepath` putanje i pretvori ga u niz stringova. Svaki string treba da bude 1 red iz fajla. Svaki red opisuje jedan objekat na slici.

    Format reda
    -----------
    class x_center y_center width height

    """
    with open(filepath, mode="r") as file:
        return file.read().splitlines()