from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.lib import pdfencrypt
from reportlab.pdfgen import canvas
from PIL import Image, ImageOps

def dag_pdf_from_image_dict(path_to_pdf, image_dict, resize_res=None):
    '''
    Make a PDF of all the images... 
    
    Parameters
    ----------
    path_to_pdf : str
        Path to the pdf to be saved
    image_dict : dict
        Dictionary containing the images to be saved. 
        Should have depth of 2
        First is rows, second is columns
    resize res:
        Image resolution to resize to 
    '''
    n_rows = len(image_dict.keys())
    n_cols = len(image_dict[list(image_dict.keys())[0]])

    # check dimensions of png files 
    # Find the dimensions of the largest image
    max_width, max_height = 0, 0
    for row in image_dict.keys():
        for col in image_dict[row].keys():
            this_path = image_dict[row][col]
            with Image.open(this_path) as img:
                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)        

    if resize_res is not None:
        max_width = resize_res[0]
        max_height = resize_res[1]

    pdf_width = max_width * n_cols
    pdf_height = max_height * n_rows
    # Create a PDF canvas
    c = canvas.Canvas(path_to_pdf, pagesize=(pdf_width, pdf_height))
    
    # Initialize
    x,y = 0, pdf_height - max_height
    
    for i_row, row in enumerate(image_dict.keys()):
        for i_col, col in enumerate(image_dict[row].keys()):
            img = Image.open(image_dict[row][col])
            if resize_res is not None:
                # img = img.resize(resize_res, Image.ANTIALIAS)  # Resize to desired resolution
                img = dag_resize_with_padding(img, resize_res)
            
            # Add resized image to PDF


            x = i_col * max_width
            y = (n_rows - i_row - 1) * max_height


            c.drawImage(ImageReader(img), x, y, max_width, max_height)  # Adjust coordinates as needed
            # c.drawImage(this_path, x, y, max_width, max_height)

            # Add subtitle text
            subtitle = f'{row} {col}'
            c.setFont("Helvetica", 10)
            c.drawString(x + 10, y + 10, subtitle)


    # Save and close the PDF
    c.showPage()
    c.save()
    


def dag_resize_with_padding(image, target_size, color=(255, 255, 255)):
    """
    Resize the image to fit within the target_size while preserving the aspect ratio.
    If necessary, pad the image with the specified color to match the target size.
    """
    original_width, original_height = image.size
    target_width, target_height = target_size

    # Calculate aspect ratios
    aspect_ratio_original = original_width / original_height
    aspect_ratio_target = target_width / target_height

    if aspect_ratio_original > aspect_ratio_target:
        # Scale based on width
        new_width = target_width
        new_height = int(new_width / aspect_ratio_original)
    else:
        # Scale based on height
        new_height = target_height
        new_width = int(new_height * aspect_ratio_original)

    # Resize the image while preserving aspect ratio
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Create a new blank image with the target size and fill it with white color
    padded_image = Image.new("RGB", (target_width, target_height), color)

    # Calculate the position to paste the resized image
    left_padding = (target_width - new_width) // 2
    top_padding = (target_height - new_height) // 2
    paste_position = (left_padding, top_padding)

    # Paste the resized image onto the padded image
    padded_image.paste(resized_image, paste_position)

    return padded_image
