# Waste Detection and Craft Recommendation App

This project focuses on the machine learning aspect of an application designed to detect waste types from images and recommend creative upcycling ideas. The application integrates with cloud computing (CC) for scalability and mobile development (MD) for accessibility.

## Machine Learning Features

- **Waste Detection**: Utilizes a trained machine learning model to classify waste into predefined categories.
- **Model Training**: Built with TensorFlow/Keras, the model is trained on a dataset of labeled waste images.
- **Custom Dataset**: Includes annotated images for various waste types, prepared using tools like LabelImg.

## Technologies Used

**Machine Learning** use TensorFlow/Keras for model training and inference.

## How to Run

### Prerequisites

- Python
- TensorFlow/Keras
- Jupyter Notebook
- Streamlit (optional for testing)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/BangkitTeam/MachineLearning
   cd repository
   ```

2. **Set Up the Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Open Jupyter Notebook**:
   Launch Jupyter Notebook and open the main.ipynb file
   Execute the cells step by step to train and test the machine learning model.

4. **Test with Streamlit** (Optional):
   Run the Streamlit app for an interactive interface:
   ```bash
   streamlit run app.py
   ```

## Dataset

The dataset includes labeled images of various waste categories [here](https://drive.google.com/drive/folders/1S8awIBJqLY7EAaBW3l4YeVSCcQDnTfi3?usp=sharing).

## Project Structure

```
MachineLearning/
|-- data/
|   |-- trash_dataset/
|-- model_result/
|-- web-streamlit/
|   |-- app.py
|-- main.ipynb
|-- README.md
|-- requirements.txt
```

## Contact

For any questions or suggestions, feel free to reach out to:
- **GitHub**:
- [reized](https://github.com/reized)
- [uchihamadara37](https://github.com/uchihamadara37)
- [azhar fikri](https://github.com/Az140304)
