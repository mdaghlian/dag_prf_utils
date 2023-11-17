from reportlab.lib.pagesizes import letter
from reportlab.lib import pdfencrypt
from reportlab.pdfgen import canvas
from PIL import Image

def dag_pdf_from_image_dict(path_to_pdf, image_dict):
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

    pdf_width = max_width * n_cols
    pdf_height = max_height * n_rows
    # Create a PDF canvas
    c = canvas.Canvas(path_to_pdf, pagesize=(pdf_width, pdf_height))
    
    # Initialize
    x,y = 0, pdf_height - max_height
    
    for i_row, row in enumerate(image_dict.keys()):
        for i_col, col in enumerate(image_dict[row].keys()):
            this_path = image_dict[row][col]
            x = i_col * max_width
            y = (n_rows - i_row - 1) * max_height
            c.drawImage(this_path, x, y, max_width, max_height)

            # Add subtitle text
            subtitle = f'{row} {col}'
            c.setFont("Helvetica", 12)
            c.drawString(x + 10, y + 10, subtitle)


    # Save and close the PDF
    c.showPage()
    c.save()
    

