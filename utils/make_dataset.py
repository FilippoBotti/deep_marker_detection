import cv2
import os
import math

def crop_image(image_path, center_x, center_y, image_name, index, output_file, real_center_x, real_center_y, folder_name):
    image = cv2.imread(image_path)
    
    top_left_x = round(center_x - 16)
    top_left_y = round(center_y - 16)
    
    cropped_image = image[top_left_y:top_left_y+32, top_left_x:top_left_x+32]
    
    if cropped_image.shape[0] < 32 or cropped_image.shape[1] < 32:
        pad_height = max(0, 32 - cropped_image.shape[0])
        pad_width = max(0, 32 - cropped_image.shape[1])
        
        cropped_image = cv2.copyMakeBorder(
            cropped_image, 0, pad_height, 0, pad_width,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
    
    os.makedirs(f"{folder_name}/images_mask", exist_ok=True)
    
    filename = f"{folder_name}/images_mask/{image_name}_{index}.jpg"
    cv2.imwrite(filename, cropped_image)
    
    output_file.write(f"{filename} {real_center_x - top_left_x -1 :.6f} {real_center_y - top_left_y -1:.6f}\n")

def main():
    folder_name ="PATH"
    image_path = "IMAGE_PATHleft000000.pgm"
    image_path2 = "IMAGE_PATHleft000001.pgm"
    image_path3 = "IMAGE_PATHleft000002.pgm"
    image_path4 = "IMAGE_PATHleft000003.pgm"
    image_path5 = "IMAGE_PATHleft000004.pgm"
    image_path6 = "IMAGE_PATHleft000005.pgm"

    image_path7 = "IMAGE_PATHright000000.pgm"
    image_path8 = "IMAGE_PATHright000001.pgm"
    image_path9 = "IMAGE_PATHright000002.pgm"
    image_path10 = "IMAGE_PATHright000003.pgm"
    image_path11 = "IMAGE_PATHright000004.pgm"
    image_path12 = "IMAGE_PATHright000005.pgm"
    imgs = [image_path,image_path2,image_path3,image_path4,image_path5,image_path6,
            image_path7,image_path8,image_path9,image_path10,image_path11,image_path12,
            ]
    output_file = open(f"{folder_name}/cropped_images.txt", "w")

    for files in imgs:
        path = files+".txt"
        with open(path, "r") as file:
            lines = file.readlines()
        
        image_name = os.path.splitext(os.path.basename(files))[0]
        print(image_name)
        
        for index, line in enumerate(lines):
            values = line.strip().split()
            center_x = float(values[2])
            center_y = float(values[3])
            crop_image(files, center_x, center_y, image_name, index, output_file, center_x, center_y, folder_name)
        
    output_file.close()

if __name__ == "__main__":
    main()
