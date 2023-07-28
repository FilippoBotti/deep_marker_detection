import cv2
import os
import math

def crop_image(image_path, center_x, center_y, image_name, index, output_file, real_center_x, real_center_y, folder_name):
    # Load the image
    image = cv2.imread(image_path)
    
    # Calculate the top-left corner coordinates for cropping
    top_left_x = round(center_x - 16)
    top_left_y = round(center_y - 16)
    
    # Crop the image
    cropped_image = image[top_left_y:top_left_y+32, top_left_x:top_left_x+32]
    
    # Check if the cropped image size is smaller than 32x32
    if cropped_image.shape[0] < 32 or cropped_image.shape[1] < 32:
        # Calculate the amount of padding needed
        pad_height = max(0, 32 - cropped_image.shape[0])
        pad_width = max(0, 32 - cropped_image.shape[1])
        
        # Add white padding to make the image 32x32
        cropped_image = cv2.copyMakeBorder(
            cropped_image, 0, pad_height, 0, pad_width,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
    
    # Create the 'images_mask' folder if it doesn't exist
    os.makedirs('images_mask', exist_ok=True)
    
    # Save the cropped image
    filename = f"{folder_name}/images_mask/{image_name}_{index}.jpg"
    cv2.imwrite(filename, cropped_image)
    
    # Write the center coordinates to the output file
    output_file.write(f"{filename} {real_center_x - top_left_x:.3f} {real_center_y - top_left_y:.3f}\n")

def main():
    folder_name = "../dataset/single"
    # Path to the image file
    image_path = "/Users/filippo/Desktop/università/visione_veicolo/progetto/0003/left000000.pgm"
    image_path2 = "/Users/filippo/Desktop/università/visione_veicolo/progetto/0001/left000001.pgm"
    # image_path3 = "/Users/filippo/Desktop/università/visione_veicolo/progetto/0003/left000002.pgm"
    # image_path4 = "/Users/filippo/Desktop/università/visione_veicolo/progetto/0003/right000000.pgm"
    # image_path5 = "/Users/filippo/Desktop/università/visione_veicolo/progetto/0003/right000001.pgm"
    # image_path6 = "/Users/filippo/Desktop/università/visione_veicolo/progetto/0003/right000002.pgm"
    #image_path3 = "/Users/filippo/Desktop/università/visione_veicolo/progetto/0002/right000000.pgm"
    imgs = [image_path, image_path2]#, image_path2, image_path3, image_path4, image_path5, image_path6]
    output_file = open(f"{folder_name}/cropped_images.txt", "w")

    # Read the file with center coordinates
    for files in imgs:
        path = files+".txt"
        with open(path, "r") as file:
            lines = file.readlines()
        
        # Get the base image name without extension
        image_name = os.path.splitext(os.path.basename(files))[0]
        print(image_name)
        
        # Create a text file to store the cropped image names and coordinates
        
        
        # Iterate over each line and crop the image
        for index, line in enumerate(lines):
            values = line.strip().split()
            # Extract the coordinates
            center_x = float(values[2])
            center_y = float(values[3])
            
            # Round the center coordinates to lower and higher values
            # lower_x = math.floor(center_x)
            # lower_y = math.floor(center_y)
            # higher_x = math.ceil(center_x)
            # higher_y = math.ceil(center_y)
            x = round(center_x)
            y = round(center_y)
            # Crop and save the images with different center approximations
            crop_image(image_path, x, y, image_name, index*4, output_file, center_x, center_y, folder_name)
            # crop_image(image_path, lower_x, higher_y, image_name, index*4+1, output_file, center_x, center_y)
            # crop_image(image_path, higher_x, lower_y, image_name, index*4+2, output_file, center_x, center_y)
            # crop_image(image_path, higher_x, higher_y, image_name, index*4+3, output_file, center_x, center_y)
        
    # Close the output file
    output_file.close()

if __name__ == "__main__":
    main()
