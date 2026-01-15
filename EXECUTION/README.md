# Image-Colorization
🎨 Image Colorization using Deep Learning (GUI-based)

This project implements **image colorization** using a **Graphical User Interface (GUI)** built with **Tkinter** and powered by a **deep learning model** from OpenCV’s Caffe framework. It allows users to upload black-and-white images and convert them to colorized versions with a simple click.

🧠 How It Works

The project uses the model from the research paper **“Colorful Image Colorization”** by *Zhang et al.*. It applies a **deep CNN** to predict the color components of a grayscale image, producing visually plausible and vibrant results.

Key Files Used:

* `colorization_deploy_v2.prototxt`: Model architecture
* `colorization_release_v2.caffemodel`: Pre-trained weights
* `pts_in_hull.npy`: Cluster centers for ab channels used during inference

📁 File Structure


📂 image-colorization-gui/
│
├── colorize_gui.py                  # GUI script for image colorization
├── colorization_deploy_v2.prototxt # Model definition
├── colorization_release_v2.caffemodel # Trained model weights
├── pts_in_hull.npy                 # Color cluster centers
├── sample_bw.jpg                   # Sample grayscale input
└── output/                         # Folder for colorized output


🔗 Download Pre-trained Model Files

You can download the necessary files from the official repository:

*  Prototxt File:
  [https://github.com/richzhang/colorization/blob/master/models/colorization\_deploy\_v2.prototxt](https://github.com/richzhang/colorization/blob/master/models/colorization_deploy_v2.prototxt)

*  Caffe Model:
  [https://github.com/richzhang/colorization/blob/master/models/colorization\_release\_v2.caffemodel](https://github.com/richzhang/colorization/blob/master/models/colorization_release_v2.caffemodel)

*  Numpy Color Clusters:
  [https://github.com/richzhang/colorization/blob/master/resources/pts\_in\_hull.npy](https://github.com/richzhang/colorization/blob/master/resources/pts_in_hull.npy)

🚀 How to Run

1. Make sure all model files are in the same directory as the script.
2. Run the GUI:

bash
python colorize_gui.py


3. A window will open where you can:

   * Upload a grayscale image
   * View the colorized output
   * Save the result

 🧰 Requirements

* Python 3.x
* OpenCV (`pip install opencv-python`)
* NumPy
* Tkinter (comes pre-installed with most Python distributions)


📌 Notes
* The model colorizes based on learned color distributions, so outputs may differ from real-life colors.
* Works best with natural and portrait images.
