import torch

import os
import shutil 
import zipfile
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from IPython.display import Image
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from xml.dom import minidom
from PIL import Image, ImageDraw



def unzip_each_invoice(src_path, dest_path):
    """
    Unzip each invoice
    """
    src_files = [os.path.join(src_path, f) for f in os.listdir(src_path) if f.endswith('.zip')]
    dest_files = [f.replace('.zip', "") for f in src_files]

    for src_file, dest_file in zip(src_files, dest_files):
        with zipfile.ZipFile(src_file, 'r') as zip_ref:
            zip_ref.extractall(dest_file)



def make_data_dir(src_path, img_dest, xml_dest):
    """
    Make data directory
    """
    src_files = [os.path.join(src_path, f) for f in os.listdir(src_path) if not f.endswith('zip')]
    
    try:
        os.mkdir(img_dest)
        os.mkdir(xml_dest)
    except Exception as e:
        print(e)

    for (dirpath, dirnames, filenames) in os.walk(src_path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                src_file = os.path.join(dirpath, filename)
                shutil.copy(src_file, img_dest)
            
            if filename.endswith('.xml'):
                src_file = os.path.join(dirpath, filename)
                shutil.copy(src_file, xml_dest)


def remove_extra_imgs(xml_folder, img_folder):
    """
    Remove images which don't have labels
    """
    xml_files = [os.path.join(xml_folder, f) for f in os.listdir(xml_folder) if f.endswith('.xml')]

    for xml_file in xml_files:
        
        image_file = xml_file.replace(xml_folder, img_folder).replace("xml", "jpg")
        if not os.path.exists(image_file):
            os.remove(xml_file)


def extract_info_from_xml(xml_file, default_depth=3):
    """
    --------
    Function to get the data from XML Annotation
    --------
    """
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                if subelem.text is not None:
                    image_size.append(int(subelem.text))
                else:
                    image_size.append(default_depth)
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = float(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict



def convert_to_yolov5(info_dict, annotations_folder):
    """
    --------
    Convert info. dict. to YOLOv5 text file
    --------
    """
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    try:
        os.mkdir(annotations_folder)
    except Exception as e:
        print(e)

    save_file_name = os.path.join(annotations_folder, info_dict["filename"].replace("jpg", "txt"))
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))






def plot_bounding_box(image, annotation_list, class_name_to_id_mapping):

    class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()



def run_test(annotations_folder, img_folder, class_name_to_id_mapping):
    """
    Run test
    """
    # Annotations
    annotations = [os.path.join(annotations_folder, f) for f in os.listdir(annotations_folder) if f.endswith('.txt')]

    # Get any random annotation file 
    annotation_file = random.choice(annotations)
    
    # Get list
    with open(annotation_file, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x ] for x in annotation_list]

    #Get the corresponding image file
    image_file = annotation_file.replace(annotations_folder, img_folder).replace("txt", "jpg")
    assert os.path.exists(image_file)

    #Load the image
    image = Image.open(image_file)

    #Plot the Bounding Box
    plot_bounding_box(image, annotation_list, class_name_to_id_mapping)





def split_data(img_folder, annotations_folder):
    """
    Split data
    """
    # Read images and annotations
    images = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('jpg')]
    annotations = [os.path.join(annotations_folder, f) for f in os.listdir(annotations_folder) if f[-3:] == "txt"]

    images.sort()
    annotations.sort()

    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, 
                                                                                    test_size = 0.1, random_state = 1)
    
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, 
                                                                                  test_size = 0.1, random_state = 1)


    return train_images, val_images, train_annotations, val_annotations, \
           val_images, test_images, val_annotations, test_annotations



def make_folders(img_folder, annotations_folder):
    """
    Make folders
    for detection
    """
    for datatype in [img_folder, annotations_folder]:
        for dataset in ['train', 'val', 'test']:
            try:
                dest = os.path.join(datatype, dataset)
                os.mkdir(dest) 
            except Exception as e:
                print(e)
                continue


def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except Exception as e:
            print(e)
            continue


def remove_files(folder, extension):
    to_remove = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extension)]
    for f in to_remove:
        os.remove(f)




if __name__ == '__main__':

    # Set random seed for reproducible results
    random.seed(1)

    # Dictionary that maps class names to IDs
    class_name_to_id_mapping = {'Vendor_Name': 0,
                                'GSTIN_vendor': 1,
                                'Invoice_Number': 2,
                                'Invoice_Date': 3, 
                                'Due_Date': 4}

    # Folder where invoice zips are located
    annotations_folder = 'annotations'
    zip_folder = 'invoices_zip'
    xml_folder = 'invoices'
    img_folder = 'images'


    # Unzip all files
    unzip_each_invoice(zip_folder, zip_folder)
    make_data_dir(zip_folder, img_folder, xml_folder)
    remove_extra_imgs(xml_folder, img_folder)

    # All files in folder
    xml_files = [os.path.join(xml_folder, f) for f in os.listdir(xml_folder) if f.endswith('.xml')]

    # Get information dictionaries
    info_dicts = [extract_info_from_xml(f) for f in xml_files]

    # Convert to YOLOv5
    for info_dict in info_dicts: 
        convert_to_yolov5(info_dict, annotations_folder)

    run_test(annotations_folder, img_folder, class_name_to_id_mapping)


    train_images, val_images, train_annotations, val_annotations, \
    val_images, test_images, val_annotations, test_annotations = split_data(img_folder, annotations_folder)

    make_folders(img_folder, annotations_folder)

    # Move the splits into their folders
    move_files_to_folder(train_images, f'{img_folder}/train')
    move_files_to_folder(val_images, f'{img_folder}/val/')
    move_files_to_folder(test_images, f'{img_folder}/test/')
    
    move_files_to_folder(train_annotations, f'{annotations_folder}/train/')
    move_files_to_folder(val_annotations, f'{annotations_folder}/val/')
    move_files_to_folder(test_annotations, f'{annotations_folder}/test/')

    remove_files(img_folder, 'jpg')
    remove_files(annotations_folder, 'txt')