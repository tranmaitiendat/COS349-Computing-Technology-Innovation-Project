# Real Estate Market Analysis Project


## Project Description
This project aims to investigate and analyze real estate prices and market performance of different types of houses in Melbourne. We use Machine Learning techniques for prediction, classification, and clustering of real estate data to gain a deeper understanding of the dynamics of the real estate market.

## Project Structure
├── README.md                                # Project description, installation, and usage guide
├── main.py                                  # Main script to run the entire process
├── preprocess.py                            # Script for data cleaning and preprocessing
├── regression.py                            # Script to build and train the regression model
├── classification.py                        # Script to build and train the classification model
├── clustering.py                            # Script to build the clustering model
├── evaluation.py                            # Script for model performance evaluation
├── notebooks/                               # Directory containing Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── regression_model.ipynb
│   ├── classification_model.ipynb
│   ├── clustering_model.ipynb
│   └── model_evaluation.ipynb
└── data/                                    # Directory containing datasets
    ├── Melbourne_housing.csv
    ├── Melbourne_housing_FULL.csv
    └── melb_data.csv


## Installation
1.Clone or download this project.
2. Navigate to the project directory:
   ```bash
  cd project_directory_name

   ```
3. Install the required libraries
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. To run the data preprocessing script, execute:
   ```bash
  python preprocess.py

   ```
   
2. To build and train the regression model, execute:
   ```bash  
   python regression.py
   ```

3. To build and train the classification model, execute:
   ```bash
   python classification.py
   ```

4. To build the clustering model, execute:
   ```bash
   python clustering.py
   ```

5. To evaluate the model, execute:
   ```bash
   python evaluation.py
   ```

6. Alternatively, you can run the entire process by executing main.py:
   ```bash
   python main.py
   ```

## References
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## 
Mai Tien Dat Tran - 104207944
Kim Thu Tran - 104061810
Trong Hoang Nam Quang - 104480444 