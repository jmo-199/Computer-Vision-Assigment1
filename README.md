# Computer-Vision-Assignment1 Image Processing

This project demonstrates various image processing techniques, including normalization, filtering, and edge detection using OpenCV and NumPy. It showcases operations like min-max normalization, Gaussian smoothing, median filtering, Sobel filtering, and noise handling.

## Introduction

This project covers image processing techniques such as:
- **Min-Max Normalization**: Adjusting pixel values between 0 and 1.
- **Down-sampling & Up-sampling**: Resizing images by reducing or increasing pixel size.
- **Gaussian Smoothing**: Applying Gaussian filters for noise reduction.
- **Median Filtering**: Removing noise using median filters.
- **Sobel Filtering**: Detecting edges using the Sobel operator.

## Requirements

- **Python 3.x**
- **NumPy** (`numpy`)
- **OpenCV** (`opencv-python`)
- **Matplotlib** (`matplotlib`)

You can install the required libraries using:

```bash
pip install numpy opencv-python matplotlib
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jmo-199/Computer-Vision-Assigment1.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Computer-Vision-Assignment1
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the project:

1. Open the Jupyter Notebook `Project_1 (3).ipynb`.
2. Execute each cell in sequence to apply image processing techniques and visualize the results.

You can also run the script directly if provided in a `.py` format.

## Features

Here’s a summary of the features demonstrated:

1. **Min-Max Normalization**: 
   Adjusts pixel intensity values to the range `[0, 1]`.
   
   Example:
   ```python
   def min_max_normalization(image):
       min_val = np.min(image)
       max_val = np.max(image)
       return (image - min_val) / (max_val - min_val)
   ```

2. **Down-sampling & Up-sampling**:
   Resizes images using various interpolation techniques.

3. **Gaussian Smoothing**: 
   Applies Gaussian filters with different kernel sizes and sigma values to smooth images.

4. **Median Filtering**: 
   Reduces noise by replacing each pixel value with the median of neighboring pixel values.

5. **Sobel Filtering**:
   Detects edges using the Sobel operator.
   
   Example:
   ```python
   sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
   sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
   gradient_x = cv2.filter2D(image, -1, sobel_x)
   gradient_y = cv2.filter2D(image, -1, sobel_y)
   ```

6. **Noise Handling**:
   Filters are applied to images with added noise to demonstrate noise reduction techniques.

## Example Workflow

This is a simplified example of how the image processing techniques are applied:

```python
# Apply Gaussian Smoothing
smoothed_image = cv2.GaussianBlur(image, (5, 5), sigmaX=1)

# Apply Median Filtering
median_filtered_image = cv2.medianBlur(image, 5)

# Apply Sobel Edge Detection
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
```

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. Fork the project.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```

3. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add feature"
   ```

4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```

5. Create a Pull Request.

## Acknowledgments

- **OpenCV**: For image processing functions.
- **NumPy**: For numerical operations.
- **Matplotlib**: For image visualization.
