# Import libraries
import cv2
import imutils
import os
# Import support functions provided by Kopernikus
from imaging_interview import compare_frames_change_detection, \
                                preprocess_image_change_detection


class SimilarPhotoRemover():
    
    def __init__(self, folder):
        # Store the path to dataset as class variable
        self.folder = folder

    def load_input(self):
        """
        This method Loads the input dataset and organizes the file names into a dictionary by camera.

        Reads the contents of the specified folder and organizes the dataset
        into a dictionary, where each camera is a key, and the corresponding values are 
        arrays of respective file names. The .DS_Store file is removed if present as it
        is not an actual photo.

        Parameters:
            None (apart from 'self', which refers to the instance of the class)

        Returns:
            None
        """

        # Remove the .DS_Store file, as it is not a photo
        try:
            os.remove(self.folder+r"\.DS_Store")
        except FileNotFoundError:
            pass

        # Dictionary to store the file names of the images with respect to the camera
        self.dataset_dict = {}

        # The dataset_dict is populated with camera names as keys. 
        # The corresponding values are arrays of respective file names
        for file_name in os.listdir(self.folder):
            camera = file_name[0:3]
            if camera not in self.dataset_dict:
                self.dataset_dict[camera] = []
            else:
                self.dataset_dict[camera].append(file_name)
        
        # Print the number of cameras from which the images are captured
        print(f"Total number of cameras = {len(self.dataset_dict.keys())}")

        # Print the number of images from each camera
        for camera in self.dataset_dict:
            print(f"Number of images from camera {camera} = \
                            {len(self.dataset_dict[camera])}")


    def identify_similar_photos(self, Gaussian_blur_radius_list_input, score_threshold_input, min_contour_area_input):
        """
        This method identifies similar photos in the dataset and populates a list of unwanted images.

        Iterates through the dataset and compares each image with the last identified unique image.
        It preprocesses each image with multi-step Gaussian blur to reduce the high frequency noise,
        and then converts them to gray scale. It uses the absolute difference between images, and
        binary thresholding to compute the contours of differences. If the score is less than a 
        specified value, the image is considered as an unwanted image.

        Parameters:
            self: instance of the class
            Gaussian_blur_radius_list_input
            score_threshold_input
            min_contour_area_input

        Returns:
            None
        """

        # List to store the names of unwanted images
        self.unwanted_images = []
        
        # Iterate through each key (i.e., camera) in the dataset_dict
        for camera in self.dataset_dict:
            num_images = len(self.dataset_dict[camera])
            
            # Initialize current and next images
            current_image = None
            next_image = None
            
            # Iterate through each image in the file
            for image_idx in range(num_images):
                
                # Read the name of the image file, and then read the image
                image_idx_name = self.dataset_dict[camera][image_idx]
                image = cv2.imread(os.path.join(self.folder, image_idx_name))

                # If cv2.imread returns None value, consider the corresponding file as unwanted
                if image is None:
                    self.unwanted_images.append(image_idx_name)
                    continue

                # If it is the first iteration, assign image to current_image and do nothing
                # Else assign image to next_image for comparision
                if current_image is None:
                    current_image = image
                    continue
                else:
                    next_image = image
                
                # If the images have different shapes, resize the next_image to the shape
                # of current_image
                if(current_image.shape != next_image.shape):
                    new_hight = current_image.shape[1]
                    new_width = current_image.shape[0]
                    new_dim = (new_hight, new_width)
                    next_image = cv2.resize(next_image, new_dim, interpolation=cv2.INTER_AREA)

                # Apply the preprocessing function on both the images
                current_image_gray = preprocess_image_change_detection(current_image, 
                                                                       gaussian_blur_radius_list = Gaussian_blur_radius_list_input)
                next_image_gray = preprocess_image_change_detection(next_image, 
                                                                    gaussian_blur_radius_list = Gaussian_blur_radius_list_input)

                # Compare the current and next gray images for change, and extract score
                score, _, _ = compare_frames_change_detection(current_image_gray, 
                                                                          next_image_gray, min_contour_area_input)

                # The next image is considered as unwanted image if the score is less than 2000
                # As long as the next unique image is not identified, we continue to compare 
                # with the current image
                if(score < score_threshold_input):
                    self.unwanted_images.append(image_idx_name)
                else:
                    # When the next unique image is identified, it becomes the current image for 
                    # future comparisions
                    current_image = next_image

        # Print the total number of unwanted images identified
        print(f"number of unwanted images = {len(self.unwanted_images)}")
  

    def remove_similar_photos(self):
        """
        This method removes unwanted images from the dataset.

        Iterates through the list of unwanted image names provided in the 'unwanted_images'
        list and attempts to remove each image from the dataset folder. If an image is missing in the
        dataset, this method handles the exception and continues to remove the rest of the images.

        Parameters:
            None (apart from 'self', which refers to the instance of the class)

        Returns:
            None
        """

        num_removed_images = 0

        # Remove each unwanted image from the dataset
        for image_name in self.unwanted_images:
            image_path = os.path.join(self.folder, image_name)

            # If an unwanted image is missing in the dataset folder, this exception handle enables the
            # program to continue to remove the rest of the images
            try:
                os.remove(image_path)
                num_removed_images += 1
            except Exception as e:
                num_removed_images -= 1
                pass
        
        # Print the total number of images removed
        print(f"Total number of removed images = {num_removed_images}")
        

if __name__ == "__main__":

    
    # Path to the dataset folder
    # *Please modify the path according to your dataset location*
    path_to_dataset = r"C:\Kopernikus automotive\remove_duplicates_task\dataset-candidates-ml\dataset"
    Gaussian_blur_radius_list_input = [5, 7]
    min_contour_area_input = 50 
    score_threshold_input = 750
    
    # Create object of the SimilarPhotoRemover Class
    my_SimilarPhotoRemover = SimilarPhotoRemover(path_to_dataset)
    
    # Call each of the methods of the object to:
    #   i. load the input
    #   ii. Identify similar photos
    #   iii. Remove the similar photos
    my_SimilarPhotoRemover.load_input()
    my_SimilarPhotoRemover.identify_similar_photos(Gaussian_blur_radius_list_input, score_threshold_input, min_contour_area_input)
    my_SimilarPhotoRemover.remove_similar_photos()
    

    
    