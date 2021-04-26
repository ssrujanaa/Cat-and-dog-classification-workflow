from numpy import save
from PIL import Image
from glob import glob

def replace_filename(input, pattern, replaceWith): 
    return input.replace(pattern, replaceWith) 
 
def rename_files(val_data):
    for image in val_data:
        im = Image.fromarray(image)
        im.save(replace_filename(image,'resized','val'), 'JPG', quality=90)
    return 0
 
def main():
    input_files = glob('resized_*.jpg')
    rename_files(input_files)
    return 0
    
if __name__ == '__main__':
    main()